namespace CalculatorWebApp.CalculatorLogic
{
	public class AddOperation : Operation
	{
		public AddOperation(Operation a, Operation b) : base(a, b)
		{
		}

        public AddOperation(double a, double b) : base(a, b)
        {
        }
        public AddOperation(Operation a, double b) : base(a, b)
        {
        }
        public AddOperation(double a, Operation b) : base(a, b)
        {
        }

        public override double Eval()
		{
			return A.Eval() + B.Eval();
		}
	}
}
