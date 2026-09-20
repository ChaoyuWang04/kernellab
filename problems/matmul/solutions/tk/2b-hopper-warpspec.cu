// warp specialization on Hopper —— 一个生产者 warpgroup 喂两个消费者
//
// 需要: modal-h100
//
// `2-hopper-tma` 用双缓冲让搬和算重叠了,但**所有 warp 还是干同一件事**:
// 一起等数据、一起发 mma。这一级按官方 educational_h100 的 level_07 + level_08
// 把分工做出来 —— 和 B200 那一列的 `4-blackwell-warpspec` 同一个思想,
// 只是 Hopper 上的形态不同。
//
// ## 三个 warpgroup
//
//     warpgroup 0        生产者:只发 TMA,decrease_registers<40>
//     warpgroup 1、2     消费者:各算一整行 A × 全部 B,increase_registers<232>
//
// **寄存器按角色重新分配**(PTX 里那条 `setmaxnreg`):生产者只按按钮,40 个够了;
// 省下来的全给消费者装累加器。Hopper 起才有这个能力。
//
// ## 顺带把「一个 block 算多块」也做了
//
//     M_BLOCK = 2    两个消费者,各负责一行 A
//     N_BLOCK = 4    每行算 4 个输出块
//
// 一个 block 产出 2×4 个 64×64 的 C 块,而不是一个。**B 的 tile 被两个消费者共用** ——
// 搬一次用两次,这是分块变大真正省掉的东西。
//
// 实现上有个取巧:4 个相邻的 B tile 在共享内存里本来就连着,直接
// `reinterpret_cast<wide_tile&>` 成一个 64×256 的宽 tile 递给 `mma_AB`,一条指令算完四块。
//
// ## H100 实测:照抄官方,**比上一级慢 3.5 倍**
//
//     case      2-hopper-tma        2b-warpspec         变化
//     4096³      319.54 TF 0.41×     90.21 TF 0.11×    0.28×
//     8192³      310.52 TF 0.39×     89.58 TF 0.11×    0.29×
//     tall       313.96 TF 0.45×     87.49 TF 0.12×    0.28×
//     deepK       47.58 TF 0.16×     14.88 TF 0.05×    0.31×
//
// 五档全过(结果对),但**每一档都慢了三倍多**。这不是笔误,是这一级的全部内容。
//
// ## 体检单给出了诊断
//
//     判定            卡在搬数据上,L2 忙到 84%
//     每线程寄存器     168
//     每块共享内存     232.6 KB   -> 每个 SM 只放得下 1 块
//     warp 位置       19% / 18%  -> 卡在寄存器上
//     **寄存器溢出**   **35,641,344 次**
//
// **三千五百万次溢出。** 累加器 `rt_fl<16, 64*N_BLOCK>` 在 N_BLOCK=4 时是每线程
// 128 个 fp32,加上宽 tile 的写回,超出了 `increase_registers<232>` 给的额度。
//
// 我把 N_BLOCK 从官方的 4 改成 2 试了一次:**4096³ 几乎没变**(89.4 → 90.2),
// deepK 快了 1.9 倍(7.98 → 14.88)。说明寄存器不是唯一的墙 —— 更硬的是那 232.6 KB
// 共享内存(`MAX_SHARED_MEMORY - 1024`,整张卡的预算),它让每个 SM 只能放 1 块,
// 占用率锁死在 18%,没有别的 warp 能盖住等待。
//
// ## 为什么留着这一级
//
// 因为它是本仓库最重要的一条纪律的实例:**不能因为它是官方的、因为它"更高级",
// 就假定它更快。**
//
// 有个事后看很明显的信号:`educational_b200/README.md` 给每一级都标了实测 TFLOPs
// (293 / 731 / 1050 / 1285),而 `educational_h100/README.md` **一个数字都没有**。
// 同一个项目,两份 README,一份带成绩一份不带 —— 这多半意味着 Hopper 那条阶梯
// 没有按同样的标准调过。
//
// 想看 warp specialization 真正值多少,去 B200 那一列的 `4-blackwell-warpspec`:
// 同一个思想,295 → 710 TFLOPS,2.4 倍。**思想没问题,这份实现的参数没调。**
//
// (真要把这一级救回来,方向是显而易见的:别申请整张卡的共享内存、把分块降到
//  每个 SM 能放下两块以上。这留给你在面板里改着玩 —— 体检单会告诉你有没有改对。)
#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;        // 几个消费者 warpgroup
constexpr int N_BLOCK = 2;        // 官方是 4 —— 实测在我们的 case 上寄存器溢出 3564 万次,见文件头
constexpr int NUM_PRODUCER_WORKERS = 4;
constexpr int NUM_CONSUMER_WORKERS = M_BLOCK * 4;
constexpr int NUM_THREADS = (NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * WARP_THREADS;

using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<BLOCK_SIZE, BLOCK_SIZE>>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(const __grid_constant__ tile_gl gA,
                   const __grid_constant__ tile_gl gB,
                   const __grid_constant__ tile_gl gC,
                   int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[2][M_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2, M_BLOCK>();
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&Bs)[2][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2, N_BLOCK>();
    
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&C_tiles)[M_BLOCK][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, M_BLOCK, N_BLOCK>();

    int tic = 0;
    int toc = 1;
    
    // Accumulator for each consumer warp group
    using wide_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE*N_BLOCK>;
    rt_fl<16, BLOCK_SIZE*N_BLOCK> C_accum;

    int row = blockIdx.y * M_BLOCK; 
    int col = blockIdx.x * N_BLOCK; 

    const int warpid = kittens::warpid();
    const int warpgroupid = warpid/4;

    // Determine type of warp group
    bool is_producer = (warpgroupid == 0);
    bool is_consumer = (warpgroupid > 0 && warpgroupid <= M_BLOCK);
    
    // Consumer index (0-based) for consumer warp groups
    int consumer_idx = is_consumer ? (warpgroupid - 1) : 0;

    __shared__ semaphore bar;
    if (threadIdx.x == 0) {
        init_semaphore(bar, 0, 1);
        tma::expect_bytes(
            bar, 
            M_BLOCK * size_bytes<typeof(As[0][0])> +
            N_BLOCK * size_bytes<typeof(Bs[0][0])>
        );
        
        // Load initial A tiles (one row per consumer)
        for (int m = 0; m < M_BLOCK; m++) {
            tma::load_async(As[tic][m], gA, {0, 0, row + m, 0}, bar);
        }
        
        // Load initial B tiles (all columns for this thread block)
        for (int n = 0; n < N_BLOCK; n++) {
            tma::load_async(Bs[tic][n], gB, {0, 0, 0, col + n}, bar);
        }
    }
    __syncthreads();

    if (is_consumer) {
        kittens::warp::zero(C_accum);
    }

    int num_tiles = K / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile, tic^=1, toc^=1) {

        wait(bar, tic);
        __syncthreads();

        if (is_producer) {
            warpgroup::decrease_registers<40>();
            if (threadIdx.x == 0 && tile+1 < num_tiles) {
                tma::expect_bytes(bar, 
                    M_BLOCK * size_bytes<typeof(As[0][0])> +
                    N_BLOCK * size_bytes<typeof(Bs[0][0])>
                );
                for (int m = 0; m < M_BLOCK; m++) {
                    tma::load_async(As[toc][m], gA, {0, 0, row + m, tile+1}, bar);
                }                
                for (int n = 0; n < N_BLOCK; n++) {
                    tma::load_async(Bs[toc][n], gB, {0, 0, tile+1, col + n}, bar);
                }
            }
        }
        else if (is_consumer) {
            warpgroup::increase_registers<232>();
            
            // Each consumer processes its assigned row of A against all columns of B
            warpgroup::mma_AB(
                C_accum,
                As[tic][consumer_idx],                // Get this consumer's A tile
                reinterpret_cast<wide_tile&>(Bs[tic][0])  // Get all B tiles as a wide tile
            );
            warpgroup::mma_async_wait();
        }
        __syncthreads();
    }

    // Store 
    if (is_consumer) {
        
        // First store the wide result to temporary tiles
        wide_tile& wide_C_temp = reinterpret_cast<wide_tile&>(C_tiles[consumer_idx][0]);
        warpgroup::store(wide_C_temp, C_accum);        
        warpgroup::sync(warpgroupid+4);
        
        // Only first warp in each consumer group stores to global memory
        if (warpid % 4 == 0) {
            for (int n = 0; n < N_BLOCK; n++) {
                tma::store_async(gC, C_tiles[consumer_idx][n], {0, 0, row + consumer_idx, col + n});
                tma::store_async_read_wait();
            }
        }
    }
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    tile_gl gA = make_gl<tile_gl>(reinterpret_cast<uint64_t>(A), 1, 1, M, K);
    tile_gl gB = make_gl<tile_gl>(reinterpret_cast<uint64_t>(B), 1, 1, K, N);
    tile_gl gC = make_gl<tile_gl>(reinterpret_cast<uint64_t>(C), 1, 1, M, N);
    dim3 block(NUM_THREADS);
    dim3 grid(N / (BLOCK_SIZE * N_BLOCK), M / (BLOCK_SIZE * M_BLOCK));   // 一个 block 产出 2×4 块
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_kernel<<<grid, block, smem, stream>>>(gA, gB, gC, M, N, K);
}
