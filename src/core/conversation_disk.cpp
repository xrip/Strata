// src/core/conversation_disk.cpp - the disk tier of the parked conversations.  See the header.
#include "strata/core/conversation_disk.hpp"

#include "strata/core/conversation_file.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <system_error>
#include <utility>

namespace strata::core {
namespace {

namespace fs = std::filesystem;

// The second half of every file's identity: with this run's random number as the first, a file another run (or
// another tier) wrote is refused on read.
constexpr uint64_t kTierConfig = 0x5354524154414449ull;   // "STRATADI"

// ids and images only: the part of a checkpoint best() matches against
ConversationCheckpoint key_of(const ConversationCheckpoint& c) {
    ConversationCheckpoint k;
    k.ids = c.ids;
    k.imgs = c.imgs;
    return k;
}

// a file of this tier (also a temporary one session_file_write left behind: ".<name>.<hex>.tmp")
bool tier_file(const std::string& name) {
    const std::string prefix = ConversationDiskTier::file_prefix();
    return name.rfind(prefix, 0) == 0 || name.rfind("." + prefix, 0) == 0;
}

void remove_file(const std::string& path) {
    std::error_code ec;
    fs::remove(fs::path(path), ec);
}

}  // namespace

ConversationDiskTier::ConversationDiskTier(ConversationDiskOptions options, std::function<Clock::time_point()> now)
    : options_(std::move(options)), now_(std::move(now)) {
    std::random_device rd;
    nonce_ = ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
}

ConversationDiskTier::~ConversationDiskTier() {
    while (!entries_.empty()) remove(0);
}

bool ConversationDiskTier::open(std::string& error) {
    open_ = false;
    if (options_.dir.empty() || options_.budget_bytes == 0) return true;   // off: nothing to open
    std::error_code ec;
    const fs::path dir(options_.dir);
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        error = "cannot create the folder " + options_.dir + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);   // the files hold conversations
    ec.clear();
#endif
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec) && tier_file(it->path().filename().string())) {
            std::error_code rm;
            fs::remove(it->path(), rm);
        }
    if (ec) {
        error = "cannot list the folder " + options_.dir + ": " + ec.message();
        return false;
    }
    open_ = true;
    return true;
}

void ConversationDiskTier::remove(size_t index, bool delete_files) {
    Entry& e = entries_.at(index);
    if (delete_files)
        for (const auto& f : e.files) remove_file(f);
    bytes_ -= e.bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
}

size_t ConversationDiskTier::expire() {
    if (!open_ || options_.ttl_s <= 0) return 0;
    const auto now = now_();
    const auto ttl = std::chrono::seconds(options_.ttl_s);
    size_t n = 0;
    while (!entries_.empty() && now - entries_.front().at >= ttl) {   // oldest first: the rest are younger
        remove(0);
        ++n;
    }
    expired_ += n;
    return n;
}

ConversationDiskTier::PutResult ConversationDiskTier::put(SavedConversation&& image) {
    PutResult r;
    SavedConversation img = std::move(image);   // freed when this returns, whatever happens
    if (!open_) { r.error = "the disk tier is off"; return r; }
    expire();
    const uint64_t estimate = img.bytes();
    if (estimate > options_.budget_bytes) {
        r.error = "larger than the disk budget";
        return r;
    }
    while (!entries_.empty() && bytes_ + estimate > options_.budget_bytes) {
        remove(0);
        ++r.evicted;
    }
    evicted_ += r.evicted;
    const auto t0 = std::chrono::steady_clock::now();
    Entry e;
    e.live = key_of(img.live);
    for (const auto& c : img.checkpoints) e.checkpoints.push_back(key_of(c));
    e.cvec = img.cvec;
    // one session file per image: the first stage's, then each later stage's (the writer does not take nested images)
    std::vector<SavedConversation> parts = std::move(img.stage_images);
    img.stage_images.clear();
    const SessionFileIdentity id{nonce_, kTierConfig};
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    wo.durable = false;   // the files live and die with this process: no flush to survive a power loss
    const uint64_t seq = ++seq_;
    auto write = [&](const SavedConversation& part, size_t k) -> bool {
        char name[96];
        std::snprintf(name, sizeof name, "%s%016llx-%llu-%zu.bin", file_prefix(), (unsigned long long) nonce_,
                      (unsigned long long) seq, k);
        const std::string path = (fs::path(options_.dir) / name).string();
        size_t written = 0;
        std::string err;
        if (!session_file_write(path, part, id, written, err, wo)) {
            r.error = err;
            remove_file(path);
            return false;
        }
        e.files.push_back(path);
        e.bytes += written;
        return true;
    };
    bool ok = write(img, 0);
    for (size_t k = 0; ok && k < parts.size(); ++k) ok = write(parts[k], k + 1);
    if (!ok) {
        for (const auto& f : e.files) remove_file(f);
        return r;
    }
    e.at = now_();
    r.bytes = e.bytes;
    bytes_ += e.bytes;
    entries_.push_back(std::move(e));
    ++written_;
    r.stored = true;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

bool ConversationDiskTier::take(size_t index, SavedConversation& out, uint64_t& bytes, std::string& error,
                                const std::function<bool(uint64_t, std::string&)>& admit) {
    out = SavedConversation{};
    bytes = 0;
    if (!open_ || index >= entries_.size()) { error = "no such entry"; return false; }
    const std::vector<std::string> files = entries_[index].files;
    remove(index, false);   // the entry goes now; its files are deleted below, after the read, whatever it gives
    const SessionFileIdentity id{nonce_, kTierConfig};
    SessionReadLimits limits;
    limits.admit = admit;
    bool ok = true;
    for (size_t k = 0; ok && k < files.size(); ++k) {
        SavedConversation part;
        size_t n = 0;
        // the RAM preflight's `admit` sees each file's peak; the images read before it are already counted by the
        // RAM figure it samples
        ok = session_file_read(files[k], id, part, n, error, limits);
        bytes += n;
        if (!ok) break;
        if (k == 0) out = std::move(part);
        else out.stage_images.push_back(std::move(part));
    }
    for (const auto& f : files) remove_file(f);
    if (!ok) { out = SavedConversation{}; return false; }
    ++read_back_;
    return true;
}

}  // namespace strata::core
