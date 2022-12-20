using System;
using MLMulticlassExperiment;

namespace MLConsumeModel
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("INICIAR PREVICIÓ");

            Console.WriteLine("Delta: ");
            string delta = Console.ReadLine();
            Console.WriteLine("Barres anteriors: ");
            string b_anteriors = Console.ReadLine();
            Console.WriteLine("Situació VWAP: ");
            string situacióVWAP = Console.ReadLine();
            


            ModelInput bookingData = CreateBookingData(delta, b_anteriors, situacióVWAP);

            ModelOutput bookingPredict = PredecirCambioReserva(bookingData);

            Console.WriteLine($"\n\nActual Reservation_status: {bookingData.BarresFutures} \nPredicted Reservation_status value {bookingPredict.Prediction} \nPredicted Reservation_status scores: [{String.Join(",", bookingPredict.Score)}]\n\n");

            Console.ReadLine();
        }

        private static ModelInput CreateBookingData(string delta, string b_anteriors, string situacióVWAP)
        {
            return new ModelInput()
            {
                Delta=Convert.ToSingle(delta),
                BarresAnteriors = b_anteriors,
                SituacioVWAP = situacióVWAP,
                BarresFutures = ""
            };
        }

        private static ModelOutput PredecirCambioReserva(ModelInput bookingData)
        {
            return ConsumeModel.Predict(bookingData);
        }
    }
}
