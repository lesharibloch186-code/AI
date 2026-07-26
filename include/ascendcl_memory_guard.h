// ============================================================================
// Memory Guard & OOM Prevention Layer for Ascend NPU
// Intercepts H2D/D2H transfers and memory allocations before CANN layer
// Provides CUDA-style error messages with clear stack traces
// Production-Ready Error Handling
// ============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <functional>
#include <stdexcept>
#include <map>
#include <chrono>
#include <iostream>

namespace ascendcl {

// Forward declarations
class Stream;

// ============================================================================
// MEMORY STATISTICS & THRESHOLDS
// ============================================================================

struct MemoryStats {
    size_t total_allocated;      // Total bytes currently allocated
    size_t peak_allocated;       // Peak allocation so far
    size_t available;            // Free bytes on device
    size_t total_device_mem;     // Total device memory
    uint32_t allocation_count;   // Number of active allocations
    uint32_t failed_attempts;    // Failed allocation attempts
    uint64_t total_h2d_bytes;    // Total H2D bytes transferred
    uint64_t total_d2h_bytes;    // Total D2H bytes transferred
};

// ============================================================================
// MEMORY EXCEPTION HIERARCHY - CUDA-style error messages
// ============================================================================

class MemoryException : public std::runtime_error {
public:
    MemoryException(const std::string& msg, const std::string& stack_trace = "")
        : std::runtime_error(msg), stack_trace_(stack_trace) {}
    
    const std::string& getStackTrace() const { return stack_trace_; }
    
private:
    std::string stack_trace_;
};

class OutOfMemoryException : public MemoryException {
public:
    OutOfMemoryException(size_t requested, size_t available,
                        const std::string& op_name,
                        const std::string& stack_trace = "")
        : MemoryException(
            std::string("CUDA_ERROR_OUT_OF_MEMORY: ") + op_name + 
            " failed to allocate " + std::to_string(requested) + " bytes. " +
            "Available: " + std::to_string(available) + " bytes.",
            stack_trace),
          requested_(requested), available_(available) {}
    
    size_t getRequestedSize() const { return requested_; }
    size_t getAvailableSize() const { return available_; }
    
private:
    size_t requested_;
    size_t available_;
};

class InvalidMemoryStateException : public MemoryException {
public:
    InvalidMemoryStateException(const std::string& detail,
                               const std::string& stack_trace = "")
        : MemoryException("CUDA_ERROR_INVALID_VALUE: " + detail, stack_trace) {}
};

class MemoryFragmentationException : public MemoryException {
public:
    MemoryFragmentationException(size_t requested, size_t available,
                                size_t largest_free_block,
                                const std::string& stack_trace = "")
        : MemoryException(
            "CUDA_ERROR_OUT_OF_MEMORY: Memory fragmentation detected. "
            "Requested: " + std::to_string(requested) + " bytes, "
            "Available: " + std::to_string(available) + " bytes, "
            "Largest free block: " + std::to_string(largest_free_block) + " bytes.",
            stack_trace),
          requested_(requested), available_(available), 
          largest_free_block_(largest_free_block) {}
    
    size_t getRequestedSize() const { return requested_; }
    size_t getAvailableSize() const { return available_; }
    size_t getLargestFreeBlock() const { return largest_free_block_; }
    
private:
    size_t requested_;
    size_t available_;
    size_t largest_free_block_;
};

// ============================================================================
// OOM HANDLER CALLBACK - User can override behavior
// ============================================================================

using OOMHandler = std::function<void(
    const std::string& op_name,      // "cudaMalloc", "MemcpyHtoD", etc.
    size_t requested_size,           // Size user requested
    size_t available_size,           // Available size at time of failure
    const std::string& details       // Additional context
)>;

// ============================================================================
// MEMORY GUARD - Intercepts allocation/transfer requests
// ============================================================================

class MemoryGuard {
public:
    // Get singleton instance
    static MemoryGuard& getInstance();
    
    // Initialize with device memory info
    void initialize(size_t total_device_memory);
    bool isInitialized() const { return initialized_; }
    
    // Enable/disable guard (default: enabled)
    void enable(bool enabled = true);
    bool isEnabled() const { return enabled_; }
    
    // Set OOM threshold (default: 95% of total device memory)
    void setOOMThreshold(float percentage);  // 0.0-1.0
    float getOOMThreshold() const { return oom_threshold_; }
    
    // Register custom OOM handler
    void setOOMHandler(const OOMHandler& handler);
    
    // Pre-flight checks (call before each operation)
    void checkAllocationFeasibility(size_t requested_size, 
                                    const std::string& op_name = "allocation");
    void checkTransferFeasibility(size_t transfer_size, const std::string& direction);
    
    // Track allocations/deallocations
    void recordAllocation(void* ptr, size_t size, bool huge_page = true);
    void recordDeallocation(void* ptr);
    
    // Update available memory (call after transfers)
    void updateAvailableMemory(size_t available);
    
    // Get current statistics
    MemoryStats getStats() const;
    
    // Reset statistics
    void resetStats();
    
    // Dump detailed memory report
    std::string dumpMemoryReport() const;
    
    // Query specific allocation info
    struct AllocationInfo {
        void* ptr;
        size_t size;
        bool huge_page;
        std::string allocation_site;
        uint64_t timestamp_ns;
    };
    
    AllocationInfo getAllocationInfo(void* ptr) const;
    std::vector<AllocationInfo> getActiveAllocations() const;
    
private:
    // Private constructor (singleton)
    MemoryGuard();
    
    // Helper methods
    void recordOOMEvent(const std::string& op_name, 
                       size_t requested, 
                       size_t available);
    std::string formatBytes(size_t bytes) const;
    
    // Member variables
    bool initialized_;
    bool enabled_;
    float oom_threshold_;                    // When to trigger OOM (0.95 = 95%)
    OOMHandler oom_handler_;                 // User callback
    
    mutable std::mutex guard_mutex_;
    
    // Statistics tracking
    MemoryStats stats_;
    
    // Active allocations tracking
    std::map<void*, AllocationInfo> allocations_;
    
    // OOM event history (last 10 events)
    struct OOMEvent {
        uint64_t timestamp_ns;
        std::string op_name;
        size_t requested;
        size_t available;
    };
    std::vector<OOMEvent> oom_events_;
};

// ============================================================================
// SAFE MEMORY WRAPPER - RAII wrapper with guard checks
// ============================================================================

class SafeDeviceMemory {
public:
    // Constructor: Allocate with guard checks
    explicit SafeDeviceMemory(size_t size, bool huge_page = true);
    
    // Destructor: Auto-cleanup with deallocation tracking
    ~SafeDeviceMemory();
    
    // Getters
    void* getData() const { return data_; }
    size_t getSize() const { return size_; }
    size_t getAlignedSize() const { return aligned_size_; }
    bool isValid() const { return data_ != nullptr; }
    
    // Prevent copying
    SafeDeviceMemory(const SafeDeviceMemory&) = delete;
    SafeDeviceMemory& operator=(const SafeDeviceMemory&) = delete;
    
    // Allow moving
    SafeDeviceMemory(SafeDeviceMemory&& other) noexcept;
    SafeDeviceMemory& operator=(SafeDeviceMemory&& other) noexcept;
    
private:
    void* data_;
    size_t size_;
    size_t aligned_size_;
    bool huge_page_;
};

// ============================================================================
// SAFE TRANSFER WRAPPERS - With guard interception
// ============================================================================

// Safe Host-to-Device transfer (with guard checks)
void SafeMemcpyHtoD(void* dst, const void* src, size_t size, 
                   Stream* stream = nullptr);

// Safe Device-to-Host transfer (with guard checks)
void SafeMemcpyDtoH(void* dst, const void* src, size_t size,
                   Stream* stream = nullptr);

// Safe Device-to-Device transfer (with guard checks)
void SafeMemcpyDtoD(void* dst, const void* src, size_t size,
                   Stream* stream = nullptr);

// ============================================================================
// DEBUG MODE - Detailed logging of all memory operations
// ============================================================================

enum class DebugLevel {
    OFF = 0,
    CRITICAL = 1,      // Only OOM events
    WARNING = 2,       // Large allocations, fragmentation
    INFO = 3,          // All alloc/dealloc events
    VERBOSE = 4        // Plus transfer details
};

void setMemoryDebugLevel(DebugLevel level);
DebugLevel getMemoryDebugLevel();

// Print detailed memory diagnostics
void printMemoryDiagnostics();

// ============================================================================
// MEMORY POOL WITH GUARD - Pre-allocate and reuse blocks
// ============================================================================

class GuardedMemoryPool {
public:
    static GuardedMemoryPool& getInstance();
    
    // Initialize pool with pre-allocated blocks
    void initialize(size_t total_pool_size, 
                   const std::vector<size_t>& block_sizes);
    
    // Allocate from pool (fast path, guarded)
    std::shared_ptr<SafeDeviceMemory> allocateFromPool(size_t size);
    
    // Get pool statistics
    struct PoolStats {
        size_t total_pool_size;
        size_t used_size;
        uint32_t available_blocks;
        uint32_t used_blocks;
    };
    PoolStats getPoolStats() const;
    
    // Release all pool memory
    void shutdown();
    
    // Check if pool is initialized
    bool isInitialized() const { return initialized_; }
    
private:
    GuardedMemoryPool() : total_size_(0), initialized_(false) {}
    
    struct PoolBlock {
        void* ptr;
        size_t size;
        bool in_use;
        uint64_t last_used_time;
    };
    
    std::vector<PoolBlock> blocks_;
    size_t total_size_;
    bool initialized_;
    mutable std::mutex pool_mutex_;
};

} // namespace ascendcl
