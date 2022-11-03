namespace CalculatorWebApp.CalculatorLogic
{
	public class MinusOperation : Operation
    {
        public MinusOperation(Operation a, Operation b) : base(a, b)
        {
        }
        public MinusOperation(double a, double b) : base(a, b)
        {
        }
        public MinusOperation(Operation a, double b) : base(a, b)
        {
        }
        public MinusOperation(double a, Operation b) : base(a, b)
        {
        }

        public override double Eval()
        {
            return A.Eval() - B.Eval();
        }
    }
}
