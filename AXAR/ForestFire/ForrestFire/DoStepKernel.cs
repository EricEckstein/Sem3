//using ILGPU;
//using ILGPU.Runtime;

//internal struct DoStepKernel
//{
//    public ArrayView2D<byte, Stride2D.DenseX> Input;
//    public ArrayView2D<byte, Stride2D.DenseX> Output;
//    public ArrayView2D<float, Stride2D.DenseX> Random;
//    public ArrayView1D<Index2D, Stride1D.Dense> Neighbours;

//    public float PGrowth;
//    public float PFire;

//    public static void Execute(Index2D idx, DoStepKernel data)
//    {
//        data.Output[idx] = data.Input[idx] switch
//        {
//            0 when data.Random[idx] < data.PGrowth => 1,
//            0 => 0,

//            1 when data.Random[idx] < data.PFire => 2,
//            1 when data.IsNeighbourBurning(idx) => 2,
//            1 => 1,

//            _ => 0
//        };
//    }

//    public bool IsNeighbourBurning(Index2D idx)
//    {
//        for(int i = 0; i < Neighbours.Length; i++) 
//        {
//            Index2D nIdx = idx + Neighbours[i];
//            if(nIdx.InBounds(Input.IntExtent) && Input[idx] == 2)
//                return true;
//        }
//        return false;
//    }
//}
