// The disk tier of --serve's parked conversations (--conversation-disk-dir): a conversation the RAM cache evicts
// (ConversationCache::set_on_evict) is written to the disk instead of dropped, kept for a limited time
// (--conversation-disk-ttl-min) within a byte budget (--conversation-disk-mib), and read back when a request continues
// it.  Each image of a layer split (the first stage's and SavedConversation::stage_images) is one session file
// (conversation_file.hpp), so the files carry the same checks as session files.  The files belong to this process
// only: their identity is a random number of this run, and the folder's files of this tier are removed at start and
// when the tier ends.  Pure host code: no CUDA.
//
// Latency: put() only queues the image; a background thread (lower CPU priority on Linux) writes it, so a
// conversation switch never waits for the disk.  A queued image that a request wants before its write started is
// handed back from RAM (no disk at all).  Until its write ends a queued image still holds its RAM: a caller short of
// RAM waits for the queue (wait_idle) before it evicts more.  Only the live state and the deepest checkpoint are
// written (the session files' rule: the earlier checkpoints serve edits further back and cost ~118 MB each).
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

struct ConversationDiskOptions {
    std::string dir;                 ///< empty: the tier is off
    uint64_t budget_bytes = 0;       ///< at most this many bytes of files; 0: off
    int64_t ttl_s = 30 * 60;         ///< a file older than this is removed (<= 0: kept until the budget evicts it)
    uint64_t min_free_bytes = 0;     ///< a write needs the file plus this much free on the disk (session preflight)
    bool background = true;          ///< write on the background thread (false: put() writes before it returns)
    /// Called with every diagnostic line (no trailing newline), from the caller's or the writer's thread.
    std::function<void(const std::string&)> log;
    /// Tests: called on the writer thread before each write starts.
    std::function<void()> before_write;
};

/// What the tier did since it started (diagnostics).
struct ConversationDiskStats {
    size_t queued = 0, written = 0, write_failed = 0, ram_hits = 0, disk_hits = 0, read_failed = 0;
    size_t expired = 0, evicted = 0, waits = 0;
    uint64_t bytes_written = 0, bytes_read = 0;
    uint64_t tokens_back = 0;                 ///< prompt tokens given back (not read again) by ram or disk hits
    double write_ms = 0, read_ms = 0, wait_ms = 0;
};

class ConversationDiskTier {
public:
    using Clock = std::chrono::steady_clock;

    explicit ConversationDiskTier(ConversationDiskOptions options, std::function<Clock::time_point()> now = Clock::now);
    ~ConversationDiskTier();
    ConversationDiskTier(const ConversationDiskTier&) = delete;
    ConversationDiskTier& operator=(const ConversationDiskTier&) = delete;

    bool enabled() const { return open_; }
    /// Creates the folder (0700 on POSIX), removes this tier's files a previous run left there and starts the
    /// writer.  False (and the tier stays off) when the folder cannot be used.
    bool open(std::string& error);

    /// Queues `image` for the disk (or writes it now without the background thread).  The image is the tier's from
    /// here on; its RAM is freed when its write ends, stored or not.  False when the tier is off.
    bool put(SavedConversation&& image);

    struct Match {
        uint64_t id = 0;        ///< the entry (stable while the writer runs; 0 = none)
        int64_t tokens = 0;
        bool live = false;
        bool queued = false;    ///< still in the write queue: take() hands it back from RAM
    };
    /// The longest prefix of `prompt` an entry holds (its live state or its checkpoint), queued or on the disk, as
    /// ConversationCache::best; ties prefer the newest entry.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::lock_guard<std::mutex> lock(m_);
        Match best;
        auto scan = [&](const Entry& e, bool queued) {
            if (e.cvec != cvec) return;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens || (n == best.tokens && n > 0 && e.id > best.id)) best = {e.id, n, live, queued};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        };
        for (const auto& e : entries_) scan(e, false);
        for (const auto& j : queue_) scan(j.entry, true);
        if (writing_) scan(writing_entry_, false);   // its file appears when the write ends: take() waits for it
        return best;
    }

    /// Hands entry `id` back in `out` (its stage images included) and removes it from the tier either way: from RAM
    /// when it is still queued, else from the disk (waiting for its write when it is being written).  `admit(need,
    /// why)` is asked before each file's payload is allocated (session_file_read's RAM preflight).  `tokens` is what
    /// the request reuses (diagnostics).  On failure `out` is empty and `error` says why.
    bool take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit = {});

    /// Whether images wait in the queue or are being written (their RAM is not back yet).
    bool pending() const;
    /// Blocks until the queue is empty and no write runs; returns the ms it waited.
    double wait_idle();

    /// Removes the entries older than the time limit; returns how many.
    size_t expire();

    size_t size() const;                    ///< entries on the disk
    uint64_t bytes() const;                 ///< their files' bytes
    ConversationDiskStats stats() const;
    /// One line of totals for the log.
    std::string summary() const;
    /// The file name prefix of this tier (tests).
    static const char* file_prefix() { return "strata-conv-"; }

private:
    struct Entry {
        uint64_t id = 0;
        std::vector<std::string> files;          ///< the first stage's image, then one per later stage
        ConversationCheckpoint live;             ///< ids and images only: what best() matches
        std::vector<ConversationCheckpoint> checkpoints;
        bool cvec = true;
        uint64_t bytes = 0;                      ///< the files' bytes (queued: the image's estimate)
        Clock::time_point at;
    };
    struct Job {
        Entry entry;
        SavedConversation image;
    };
    void writer();
    void write(Job& job);                                    // without the lock
    void remove_locked(size_t index, bool delete_files);     // m_ held
    size_t expire_locked();                                  // m_ held
    void log(const std::string& line) const;

    ConversationDiskOptions options_;
    std::function<Clock::time_point()> now_;
    bool open_ = false;
    uint64_t nonce_ = 0, next_id_ = 0, bytes_ = 0;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Entry> entries_;   ///< on the disk, oldest first
    std::deque<Job> queue_;       ///< waiting for the writer, oldest first
    bool writing_ = false;        ///< the writer holds a job (writing_entry_ is its keys)
    Entry writing_entry_;
    bool stop_ = false;
    std::thread thread_;
    ConversationDiskStats stats_;
};

}  // namespace strata::core
