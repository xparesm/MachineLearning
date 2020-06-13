using System;

namespace MLMulticlassExperiment
{
    class Program
    {
        static void Main(string[] args)
        {
            //Run this project to create the experiment and the model
            Console.WriteLine("Hello ML World!");

            ModelBuilder.CreateExperiment();

            Console.ReadKey();
        }
    }
}
