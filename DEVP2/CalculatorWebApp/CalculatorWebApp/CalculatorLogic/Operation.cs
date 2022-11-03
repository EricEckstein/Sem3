namespace CalculatorWebApp.CalculatorLogic
{
	public abstract class Operation
	{
		public Operation A { get; set; }
		public Operation B { get; set; }

        public Operation(Operation a, Operation b)
		{
			A = a;
			B = b;
		}

        public Operation(double a, double b)
        {
            A = new NumberOperation(a);
            B = new NumberOperation(b);
        }

        public Operation(Operation a, double b)
        {
            A = a;
            B = new NumberOperation(b);
        }

        public Operation(double a, Operation b)
        {
            A = new NumberOperation(a);
            B = b;
        }

        public abstract double Eval();
	}
}
