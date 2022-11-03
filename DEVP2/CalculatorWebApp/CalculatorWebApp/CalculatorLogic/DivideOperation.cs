namespace CalculatorWebApp.CalculatorLogic
{
	public class DivideOperation : Operation
	{
		public DivideOperation(Operation a, Operation b) : base(a, b)
		{
		}
        public DivideOperation(double a, double b) : base(a, b)
        {
        }
        public DivideOperation(Operation a, double b) : base(a, b)
        {
        }
        public DivideOperation(double a, Operation b) : base(a, b)
        {
        }

        public override double Eval()
		{
			return A.Eval() / B.Eval();
		}
	}
}
