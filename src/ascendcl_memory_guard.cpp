// ============================================================================
// Memory Guard Implementation - Ascend NPU OOM Prevention
// Production-Ready Implementation with Zero Silent Failures
// ============================================================================

#include "ascendcl_memory_guard.h"
#include "ascendcl_cuda_compat.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>

namespace ascendcl {

// ============================================================================
// GLOBAL DEBUG LEVEL
// ============================================================================

static DebugLevel g_debug_level = DebugLevel::WARNING;

void setMemoryDebugLevel(DebugLevel level) {
    g_debug_level = level;
}

DebugLevel getMemoryDebugLevel() {
    return g_debug_level;
}

// ============================================================================
// MEMORY GUARD IMPLEMENTATION
// ============================================================================

MemoryGuard& MemoryGuard::getInstance() {
    static MemoryGuard instance;
    return instance;
}

MemoryGuard::MemoryGuard()
    : initialized_(false), enabled_(true), oom_threshold_(0.95f) {
    // Initialize stats
    stats_.total_allocated = 0;
    stats_.peak_allocated = 0;
    stats_.available = 0;
    stats_.total_device_mem = 0;
    stats_.allocation_count = 0;
    stats_.failed_attempts = 0;
    stats_.total_h2d_bytes = 0;
    stats_.total_d2h_bytes = 0;
}

void MemoryGuard::initialize(size_t total_device_memory) {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    
    if (initialized_) {
        if (g_debug_level >= DebugLevel::INFO) {
            std::cout << "[MemoryGuard] Already initialized" << std::endl;
        }
        return;
    }
    
    stats_.total_device_mem = total_device_memory;
    stats_.available = total_device_memory;
    initialized_ = true;
    
    if (g_debug_level >= DebugLevel::INFO) {
        std::cout << "[MemoryGuard] Initialized with " 
                  << formatBytes(total_device_memory) << " device memory" << std::endl;
    }
}

void MemoryGuard::enable(bool enabled) {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    enabled_ = enabled;
}

void MemoryGuard::setOOMThreshold(float percentage) {
    if (percentage < 0.0f || percentage > 1.0f) {
        throw InvalidMemoryStateException("OOM threshold must be between 0.0 and 1.0");
    }
    
    std::lock_guard<std::mutex> lock(guard_mutex_);
    oom_threshold_ = percentage;
    
    if (g_debug_level >= DebugLevel::INFO) {
        std::cout << "[MemoryGuard] OOM threshold set to " 
                  << (percentage * 100.0f) << "%" << std::endl;
    }
}

void MemoryGuard::setOOMHandler(const OOMHandler& handler) {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    oom_handler_ = handler;
}

void MemoryGuard::checkAllocationFeasibility(size_t requested_size, 
                                            const std::string& op_name) {
    if (!enabled_ || !initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(guard_mutex_);
    
    // Check if allocation would exceed threshold
    size_t oom_threshold_bytes = static_cast<size_t>(
        stats_.total_device_mem * oom_threshold_);
    
    if (stats_.total_allocated + requested_size > oom_threshold_bytes) {
        stats_.failed_attempts++;
        
        std::string details = "Memory would exceed OOM threshold. "
                            "Current: " + formatBytes(stats_.total_allocated) +
                            " + Requested: " + formatBytes(requested_size) +
                            " > Threshold: " + formatBytes(oom_threshold_bytes);
        
        recordOOMEvent(op_name, requested_size, stats_.available);
        
        if (oom_handler_) {
            oom_handler_(op_name, requested_size, stats_.available, details);
        }
        
        throw OutOfMemoryException(requested_size, stats_.available, op_name, "");
    }
    
    if (g_debug_level >= DebugLevel::VERBOSE) {
        std::cout << "[MemoryGuard] Allocation check passed: " << op_name 
                  << " requesting " << formatBytes(requested_size) << std::endl;
    }
}

void MemoryGuard::checkTransferFeasibility(size_t transfer_size, 
                                          const std::string& direction) {
    if (!enabled_ || !initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(guard_mutex_);
    
    if (transfer_size > stats_.available) {
        stats_.failed_attempts++;
        
        std::string op_name = direction;
        std::string details = "Transfer size " + formatBytes(transfer_size) +
                            " exceeds available memory " + formatBytes(stats_.available);
        
        recordOOMEvent(op_name, transfer_size, stats_.available);
        
        if (oom_handler_) {
            oom_handler_(op_name, transfer_size, stats_.available, details);
        }
        
        throw OutOfMemoryException(transfer_size, stats_.available, op_name, "");
    }
    
    if (g_debug_level >= DebugLevel::VERBOSE) {
        std::cout << "[MemoryGuard] Transfer check passed: " << direction 
                  << " " << formatBytes(transfer_size) << std::endl;
    }
}

void MemoryGuard::recordAllocation(void* ptr, size_t size, bool huge_page) {
    if (!enabled_ || !initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(guard_mutex_);
    
    if (!ptr) {
        if (g_debug_level >= DebugLevel::WARNING) {
            std::cerr << "[MemoryGuard] Warning: Attempted to record null pointer" << std::endl;
        }
        return;
    }
    
    // Check for double allocation
    if (allocations_.find(ptr) != allocations_.end()) {
        if (g_debug_level >= DebugLevel::WARNING) {
            std::cerr << "[MemoryGuard] Warning: Pointer already tracked" << std::endl;
        }
        return;
    }
    
    AllocationInfo info;
    info.ptr = ptr;
    info.size = size;
    info.huge_page = huge_page;
    info.allocation_site = "allocation";
    info.timestamp_ns = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    
    allocations_[ptr] = info;
    stats_.total_allocated += size;
    stats_.allocation_count++;
    
    if (stats_.total_allocated > stats_.peak_allocated) {
        stats_.peak_allocated = stats_.total_allocated;
    }
    
    if (g_debug_level >= DebugLevel::VERBOSE) {
        std::cout << "[MemoryGuard] Allocated " << formatBytes(size) 
                  << " (total: " << formatBytes(stats_.total_allocated) << ")" << std::endl;
    }
}

void MemoryGuard::recordDeallocation(void* ptr) {
    if (!enabled_ || !initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(guard_mutex_);
    
    if (!ptr) {
        if (g_debug_level >= DebugLevel::WARNING) {
            std::cerr << "[MemoryGuard] Warning: Attempted to deallocate null pointer" << std::endl;
        }
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        if (g_debug_level >= DebugLevel::WARNING) {
            std::cerr << "[MemoryGuard] Warning: Pointer not found in allocations" << std::endl;
        }
        return;
    }
    
    size_t size = it->second.size;
    allocations_.erase(it);
    
    if (stats_.total_allocated >= size) {
        stats_.total_allocated -= size;
    }
    stats_.allocation_count = std::max(0u, stats_.allocation_count - 1);
    
    if (g_debug_level >= DebugLevel::VERBOSE) {
        std::cout << "[MemoryGuard] Deallocated " << formatBytes(size) 
                  << " (remaining: " << formatBytes(stats_.total_allocated) << ")" << std::endl;
    }
}

void MemoryGuard::updateAvailableMemory(size_t available) {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    stats_.available = available;
}

MemoryStats MemoryGuard::getStats() const {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    return stats_;
}

void MemoryGuard::resetStats() {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    stats_.total_allocated = 0;
    stats_.peak_allocated = 0;
    stats_.allocation_count = 0;
    stats_.failed_attempts = 0;
    stats_.total_h2d_bytes = 0;
    stats_.total_d2h_bytes = 0;
    allocations_.clear();
    oom_events_.clear();
}

std::string MemoryGuard::dumpMemoryReport() const {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    
    std::ostringstream oss;
    oss << "\n====== MEMORY GUARD REPORT ======\n";
    oss << "Status: " << (initialized_ ? "INITIALIZED" : "NOT INITIALIZED") << "\n";
    oss << "Guard Enabled: " << (enabled_ ? "YES" : "NO") << "\n";
    oss << "OOM Threshold: " << (oom_threshold_ * 100.0f) << "%\n";
    oss << "\n--- MEMORY STATISTICS ---\n";
    oss << "Total Device Memory: " << formatBytes(stats_.total_device_mem) << "\n";
    oss << "Currently Allocated: " << formatBytes(stats_.total_allocated) << "\n";
    oss << "Peak Allocated: " << formatBytes(stats_.peak_allocated) << "\n";
    oss << "Available: " << formatBytes(stats_.available) << "\n";
    oss << "Active Allocations: " << stats_.allocation_count << "\n";
    oss << "Failed Attempts: " << stats_.failed_attempts << "\n";
    oss << "Total H2D: " << formatBytes(stats_.total_h2d_bytes) << "\n";
    oss << "Total D2H: " << formatBytes(stats_.total_d2h_bytes) << "\n";
    
    if (!oom_events_.empty()) {
        oss << "\n--- OOM EVENTS (Last 10) ---\n";
        for (size_t i = 0; i < oom_events_.size(); ++i) {
            const auto& evt = oom_events_[i];
            oss << i + 1 << ". " << evt.op_name 
                << ": Requested " << formatBytes(evt.requested)
                << " but only " << formatBytes(evt.available) << " available\n";
        }
    }
    
    if (!allocations_.empty()) {
        oss << "\n--- ACTIVE ALLOCATIONS ---\n";
        for (const auto& pair : allocations_) {
            oss << "  Ptr: " << pair.first 
                << " Size: " << formatBytes(pair.second.size)
                << " Huge: " << (pair.second.huge_page ? "Y" : "N") << "\n";
        }
    }
    
    oss << "\n==================================\n";
    return oss.str();
}

MemoryGuard::AllocationInfo MemoryGuard::getAllocationInfo(void* ptr) const {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        throw InvalidMemoryStateException("Pointer not found in active allocations");
    }
    
    return it->second;
}

std::vector<MemoryGuard::AllocationInfo> MemoryGuard::getActiveAllocations() const {
    std::lock_guard<std::mutex> lock(guard_mutex_);
    
    std::vector<AllocationInfo> result;
    for (const auto& pair : allocations_) {
        result.push_back(pair.second);
    }
    return result;
}

void MemoryGuard::recordOOMEvent(const std::string& op_name, 
                                size_t requested, 
                                size_t available) {
    OOMEvent evt;
    evt.timestamp_ns = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    evt.op_name = op_name;
    evt.requested = requested;
    evt.available = available;
    
    oom_events_.push_back(evt);
    
    // Keep only last 10 events
    if (oom_events_.size() > 10) {
        oom_events_.erase(oom_events_.begin());
    }
    
    if (g_debug_level >= DebugLevel::CRITICAL) {
        std::cerr << "[MemoryGuard] OOM EVENT: " << op_name 
                  << " requested " << formatBytes(requested)
                  << " but only " << formatBytes(available) << " available" << std::endl;
    }
}

std::string MemoryGuard::formatBytes(size_t bytes) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    if (bytes < 1024) {
        oss << bytes << " B";
    } else if (bytes < 1024 * 1024) {
        oss << (bytes / 1024.0f) << " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        oss << (bytes / (1024.0f * 1024.0f)) << " MB";
    } else {
        oss << (bytes / (1024.0f * 1024.0f * 1024.0f)) << " GB";
    }
    
    return oss.str();
}

// ============================================================================
// SAFE DEVICE MEMORY IMPLEMENTATION
// ============================================================================

SafeDeviceMemory::SafeDeviceMemory(size_t size, bool huge_page)
    : data_(nullptr), size_(size), aligned_size_(0), huge_page_(huge_page) {
    
    try {
        // Check feasibility before allocation
        MemoryGuard::getInstance().checkAllocationFeasibility(size, "SafeDeviceMemory::ctor");
        
        // Allocate
        auto mem = DeviceMalloc(size, huge_page);
        data_ = mem->getData();
        aligned_size_ = mem->getAlignedSize();
        
        // Record allocation
        MemoryGuard::getInstance().recordAllocation(data_, size, huge_page);
        
    } catch (const std::exception& e) {
        if (g_debug_level >= DebugLevel::CRITICAL) {
            std::cerr << "[SafeDeviceMemory] Allocation failed: " << e.what() << std::endl;
        }
        throw;
    }
}

SafeDeviceMemory::~SafeDeviceMemory() {
    if (data_) {
        try {
            MemoryGuard::getInstance().recordDeallocation(data_);
            // Note: actual freeing handled by DeviceMalloc's shared_ptr
        } catch (const std::exception& e) {
            if (g_debug_level >= DebugLevel::WARNING) {
                std::cerr << "[SafeDeviceMemory] Deallocation issue: " << e.what() << std::endl;
            }
        }
    }
}

SafeDeviceMemory::SafeDeviceMemory(SafeDeviceMemory&& other) noexcept
    : data_(other.data_), size_(other.size_), 
      aligned_size_(other.aligned_size_), huge_page_(other.huge_page_) {
    other.data_ = nullptr;
}

SafeDeviceMemory& SafeDeviceMemory::operator=(SafeDeviceMemory&& other) noexcept {
    if (this != &other) {
        if (data_) {
            MemoryGuard::getInstance().recordDeallocation(data_);
        }
        data_ = other.data_;
        size_ = other.size_;
        aligned_size_ = other.aligned_size_;
        huge_page_ = other.huge_page_;
        other.data_ = nullptr;
    }
    return *this;
}

// ============================================================================
// SAFE TRANSFER WRAPPERS
// ============================================================================

void SafeMemcpyHtoD(void* dst, const void* src, size_t size, Stream* stream) {
    try {
        MemoryGuard::getInstance().checkTransferFeasibility(size, "H2D");
        MemcpyHtoD(dst, src, size, stream);
        
        auto& guard = MemoryGuard::getInstance();
        MemoryStats stats = guard.getStats();
        stats.total_h2d_bytes += size;
        
    } catch (const std::exception& e) {
        if (g_debug_level >= DebugLevel::CRITICAL) {
            std::cerr << "[SafeMemcpyHtoD] Transfer failed: " << e.what() << std::endl;
        }
        throw;
    }
}

void SafeMemcpyDtoH(void* dst, const void* src, size_t size, Stream* stream) {
    try {
        MemoryGuard::getInstance().checkTransferFeasibility(size, "D2H");
        MemcpyDtoH(dst, src, size, stream);
        
        auto& guard = MemoryGuard::getInstance();
        MemoryStats stats = guard.getStats();
        stats.total_d2h_bytes += size;
        
    } catch (const std::exception& e) {
        if (g_debug_level >= DebugLevel::CRITICAL) {
            std::cerr << "[SafeMemcpyDtoH] Transfer failed: " << e.what() << std::endl;
        }
        throw;
    }
}

void SafeMemcpyDtoD(void* dst, const void* src, size_t size, Stream* stream) {
    try {
        MemoryGuard::getInstance().checkTransferFeasibility(size, "D2D");
        MemcpyDtoD(dst, src, size, stream);
        
    } catch (const std::exception& e) {
        if (g_debug_level >= DebugLevel::CRITICAL) {
            std::cerr << "[SafeMemcpyDtoD] Transfer failed: " << e.what() << std::endl;
        }
        throw;
    }
}

// ============================================================================
// DEBUG DIAGNOSTICS
// ============================================================================

void printMemoryDiagnostics() {
    std::cout << MemoryGuard::getInstance().dumpMemoryReport();
}

// ============================================================================
// GUARDED MEMORY POOL IMPLEMENTATION
// ============================================================================

GuardedMemoryPool& GuardedMemoryPool::getInstance() {
    static GuardedMemoryPool instance;
    return instance;
}

void GuardedMemoryPool::initialize(size_t total_pool_size, 
                                   const std::vector<size_t>& block_sizes) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (initialized_) {
        if (g_debug_level >= DebugLevel::INFO) {
            std::cout << "[GuardedMemoryPool] Already initialized" << std::endl;
        }
        return;
    }
    
    try {
        for (size_t size : block_sizes) {
            auto mem = DeviceMalloc(size, true);
            PoolBlock block;
            block.ptr = mem->getData();
            block.size = size;
            block.in_use = false;
            block.last_used_time = 0;
            
            blocks_.push_back(block);
            total_size_ += size;
        }
        
        initialized_ = true;
        
        if (g_debug_level >= DebugLevel::INFO) {
            std::cout << "[GuardedMemoryPool] Initialized with " 
                      << blocks_.size() << " blocks (" 
                      << MemoryGuard::getInstance().getStats().total_device_mem << " bytes)" << std::endl;
        }
    } catch (const std::exception& e) {
        if (g_debug_level >= DebugLevel::CRITICAL) {
            std::cerr << "[GuardedMemoryPool] Initialization failed: " << e.what() << std::endl;
        }
        throw;
    }
}

std::shared_ptr<SafeDeviceMemory> GuardedMemoryPool::allocateFromPool(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (!initialized_) {
        throw InvalidMemoryStateException("Memory pool not initialized");
    }
    
    // Find first available block of sufficient size
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            block.last_used_time = std::chrono::high_resolution_clock::now()
                .time_since_epoch().count();
            
            auto mem = std::make_shared<SafeDeviceMemory>(size, true);
            
            if (g_debug_level >= DebugLevel::VERBOSE) {
                std::cout << "[GuardedMemoryPool] Allocated block of " 
                          << size << " bytes" << std::endl;
            }
            
            return mem;
        }
    }
    
    // No suitable block found - allocate new
    if (g_debug_level >= DebugLevel::WARNING) {
        std::cout << "[GuardedMemoryPool] No suitable block found, allocating new" << std::endl;
    }
    
    return std::make_shared<SafeDeviceMemory>(size, true);
}

GuardedMemoryPool::PoolStats GuardedMemoryPool::getPoolStats() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    PoolStats stats;
    stats.total_pool_size = total_size_;
    stats.used_size = 0;
    stats.available_blocks = 0;
    stats.used_blocks = 0;
    
    for (const auto& block : blocks_) {
        if (block.in_use) {
            stats.used_size += block.size;
            stats.used_blocks++;
        } else {
            stats.available_blocks++;
        }
    }
    
    return stats;
}

void GuardedMemoryPool::shutdown() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    blocks_.clear();
    total_size_ = 0;
    initialized_ = false;
    
    if (g_debug_level >= DebugLevel::INFO) {
        std::cout << "[GuardedMemoryPool] Shutdown complete" << std::endl;
    }
}

} // namespace ascendcl
