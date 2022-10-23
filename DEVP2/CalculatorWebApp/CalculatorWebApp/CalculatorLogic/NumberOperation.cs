namespace CalculatorWebApp.CalculatorLogic
{
	public class NumberOperation : Operation
	{
		double d;

		public NumberOperation(double d) : base(d,0)
		{
		}

		public override double Eval()
		{
			return d;
		}
	}
}
