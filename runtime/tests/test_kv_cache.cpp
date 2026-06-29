/**
 * KV Cache 单元测试 (可在 PC 上运行, 无 NPU 依赖)
 */
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../kv_cache.hpp"

using namespace rkmrt2;

int main() {
    printf("=== KV Cache Test ===\n");

    // 配置
    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.max_kv_len = 8;        // 小窗口用于测试
    cfg.dim_per_head = 64;
    cfg.cond_len = 10;
    cfg.dtype_size = 2;        // FP16

    KVCacheManager mgr(cfg);
    assert(mgr.num_layers() == 2);

    // 测试写入
    int byte_per_head = cfg.dim_per_head * cfg.dtype_size;
    std::vector<uint8_t> dummy_k(cfg.num_heads * byte_per_head, 0x42);
    std::vector<uint8_t> dummy_v(cfg.num_heads * byte_per_head, 0x24);

    for (int step = 0; step < 20; step++) {
        mgr.update_self_kv(0, dummy_k.data(), dummy_v.data());
        mgr.update_self_kv(1, dummy_k.data(), dummy_v.data());

        int expected_len = (step + 1 < cfg.max_kv_len) ? step + 1 : cfg.max_kv_len;
        assert(mgr.layer(0).current_length() == expected_len);
    }

    // 测试重置
    mgr.reset_self();
    assert(mgr.layer(0).current_length() == 0);

    // 测试 cross-attn KV
    std::vector<uint8_t> cross_k(cfg.num_heads * cfg.cond_len * byte_per_head, 0x11);
    std::vector<uint8_t> cross_v(cfg.num_heads * cfg.cond_len * byte_per_head, 0x22);
    mgr.set_cross_kv(0, cross_k.data(), cross_v.data());

    printf("  [PASS] All KV cache tests\n");
    printf("=== Done ===\n");
    return 0;
}
