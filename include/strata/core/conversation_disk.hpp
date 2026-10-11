// The disk tier of --serve's parked conversations (--conversation-disk-dir): a conversation the RAM cache evicts
// (ConversationCache::set_on_evict) is written to the disk instead of dropped, kept for a limited time
// (--conversation-disk-ttl-min) within a byte budget (--conversation-disk-mib), and read back when a request continues
// it.  Each image of a layer split (the first stage's and SavedConversation::stage_images) is one session file
// (conversation_file.hpp), so the files carry the same checks as session files.  The files belong to this process
// only: their identity is a random number of this run, and the folder's files of this tier are removed at start and
// when the tier ends.  Pure host code: no CUDA.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

struct ConversationDiskOptions {
    std::string dir;                 ///< empty: the tier is off
    uint64_t budget_bytes = 0;       ///< at most this many bytes of files; 0: off
    int64_t ttl_s = 30 * 60;         ///< a file older than this is removed (<= 0: kept until the budget evicts it)
    uint64_t min_free_bytes = 0;     ///< a write needs the file plus this much free on the disk (session preflight)
};

class ConversationDiskTier {
public:
    using Clock = std::chrono::steady_clock;

    explicit ConversationDiskTier(ConversationDiskOptions options, std::function<Clock::time_point()> now = Clock::now);
    ~ConversationDiskTier();
    ConversationDiskTier(const ConversationDiskTier&) = delete;
    ConversationDiskTier& operator=(const ConversationDiskTier&) = delete;

    bool enabled() const { return open_; }
    /// Creates the folder (0700 on POSIX) and removes this tier's files a previous run left there.  False (and the
    /// tier stays off) when the folder cannot be used.
    bool open(std::string& error);

    struct PutResult {
        bool stored = false;
        uint64_t bytes = 0;      ///< what the files took
        double ms = 0;
        size_t evicted = 0;      ///< older entries removed for the budget
        std::string error;       ///< why it was not stored
    };
    /// Writes `image` (and its stage images) to the disk; the image's RAM is freed when this returns, stored or not.
    PutResult put(SavedConversation&& image);

    /// The longest prefix of `prompt` an entry on the disk holds (its live state or one of its checkpoints), as
    /// ConversationCache::best; ties prefer the newest entry.
    template<class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                                  bool cvec) const {
        ConversationCache::Match best;
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    /// Reads entry `index` back into `out` (its stage images included) and removes it from the disk either way.
    /// `admit(need_bytes, why)` is asked before each file's payload is allocated (the RAM preflight of
    /// session_file_read).  On failure `out` is empty and `error` says why.
    bool take(size_t index, SavedConversation& out, uint64_t& bytes, std::string& error,
              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit = {});

    /// Removes the entries older than the time limit; returns how many.
    size_t expire();

    size_t size() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    size_t written() const { return written_; }
    size_t read_back() const { return read_back_; }
    size_t expired() const { return expired_; }
    size_t evicted() const { return evicted_; }
    /// The file name prefix of this tier (tests).
    static const char* file_prefix() { return "strata-conv-"; }

private:
    struct Entry {
        std::vector<std::string> files;          ///< the first stage's image, then one per later stage
        ConversationCheckpoint live;             ///< ids and images only: what best() matches
        std::vector<ConversationCheckpoint> checkpoints;
        bool cvec = true;
        uint64_t bytes = 0;
        Clock::time_point at;
    };
    void remove(size_t index, bool delete_files = true);

    ConversationDiskOptions options_;
    std::function<Clock::time_point()> now_;
    bool open_ = false;
    uint64_t nonce_ = 0, seq_ = 0, bytes_ = 0;
    size_t written_ = 0, read_back_ = 0, expired_ = 0, evicted_ = 0;
    std::deque<Entry> entries_;   ///< oldest first
};

}  // namespace strata::core
