namespace CalculatorWebApp.CalculatorLogic
{
    public class MultiplyOperation : Operation
    {
        public MultiplyOperation(Operation a, Operation b) : base(a, b)
        {
        }
        public MultiplyOperation(double a, double b) : base(a, b)
        {
        }
        public MultiplyOperation(Operation a, double b) : base(a, b)
        {
        }
        public MultiplyOperation(double a, Operation b) : base(a, b)
        {
        }

        public override double Eval()
        {
            return A.Eval() * B.Eval();
        }
    }
}