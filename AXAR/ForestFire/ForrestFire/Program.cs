//using ILGPU;
//using ILGPU.Runtime;
//using ILGPU.Runtime.Cuda;
//using System;

//public class Program
//{
//    private DoStepKernel _doStepKernel;

//    protected class Neighbourhood
//    {
//        int[,] buff = new int[1024, 1024];

//        protected Index2D[] ToIndices()
//        {
//            return buff.getas

//        }
//    }

//    public void DoStep(float pGrowth, float pFire, Neighbourhood neighbourhood)
//    {
//        _rng.FillUniform(_rndBuffer.View);

//        Index2D[] nIdices = neighbourhood.ToIndices();
//        using MemoryBuffer1D<Index2D, Stride1D.Dense> neighbourBuffer = _accelerator.Allocate1D<Index2D>(nIdices.Length);
//        neighbourBuffer.CopyFromCPU(nIdices);

//        _doStepKernel(_outputBuffer.Extent.ToIntIndex(), new DoStepKernel
//        {
//            Input = _inputBuffer.View,
//            Output = _outputBuffer.View,
//            Random = _rndBuffer.View,
//            PGrowth = 0.2,
//            PFire = 0.1,
//            Neighbours = neighbourBuffer
//        });

//        (_inputBuffer, _outputBuffer) = (_outputBuffer, _inputBuffer);
//    }

static void Main()
{ }
//    static void Main()
//    {
//        // Initialize ILGPU.

//        Context context = Context.Create(builder => builder.Cuda());
//        Accelerator accelerator = context.GetPreferredDevice(preferCPU: false)
//                                  .CreateAccelerator(context);

//        // Load the data.
//        MemoryBuffer1D<int, Stride1D.Dense> deviceData = accelerator.Allocate1D(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//        MemoryBuffer1D<int, Stride1D.Dense> deviceOutput = accelerator.Allocate1D<int>(10_000);

//        // load / precompile the kernel
//        Action<Index1D, ArrayView<int>, ArrayView<int>> loadedKernel =
//            accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(Kernel);

//        // finish compiling and tell the accelerator to start computing the kernel
//        loadedKernel((int)deviceOutput.Length, deviceData.View, deviceOutput.View);

//        // wait for the accelerator to be finished with whatever it's doing
//        // in this case it just waits for the kernel to finish.
//        accelerator.Synchronize();

//        // moved output data from the GPU to the CPU for output to console
//        int[] hostOutput = deviceOutput.GetAsArray1D();

//        for (int i = 0; i < 50; i++)
//        {
//            Console.Write(hostOutput[i]);
//            Console.Write(" ");
//        }

//        accelerator.Dispose();
//        context.Dispose();
//    }
//}
