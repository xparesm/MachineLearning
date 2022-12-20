using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;


namespace MLMulticlassExperiment
{
    public static class ModelBuilder
    {
        private static string TRAIN_DATA_FILEPATH = @"C:\Users\xpare\source\repos\MLMulticlassExperiment\MLMulticlassExperiment\data\DataTrainer.csv";
        private static string TEST_DATA_FILEPATH = @"C:\Users\xpare\source\repos\MLMulticlassExperiment\MLMulticlassExperiment\data\DataTest.csv";
        private static string MODEL_FILEPATH = @"C:\Users\xpare\source\repos\MLMulticlassExperiment\MLMulticlassExperiment\data\MLModel.zip";

        private static MLContext mlContext = new MLContext();
        private static uint ExperimentTime = 12;

        public static void CreateExperiment()
        {
            // Load Data
            var tmpPath = GetAbsolutePath(TRAIN_DATA_FILEPATH);
            var testPath = GetAbsolutePath(TEST_DATA_FILEPATH);

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: tmpPath,
                                            hasHeader: true,
                                            separatorChar: ';',
                                            allowQuoting: true,
                                            allowSparse: false);

            IDataView testingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: testPath,
                                            hasHeader: true,
                                            separatorChar: ';',
                                            allowQuoting: true,
                                            allowSparse: false
                                            );

            // STEP 2: Run AutoML experiment
            Console.WriteLine($"Running AutoML Multiclass classification experiment for {ExperimentTime} seconds...");
            ExperimentResult<MulticlassClassificationMetrics> experimentResult = mlContext.Auto()
                .CreateMulticlassClassificationExperiment(ExperimentTime)
                .Execute(trainingDataView, labelColumnName: "b_futures");
            
            // STEP 3: Print metric from the best model
            RunDetail<MulticlassClassificationMetrics> bestRun = experimentResult.BestRun;
            Console.WriteLine($"Total models produced: {experimentResult.RunDetails.Count()}");
            Console.WriteLine($"Best model's trainer: {bestRun.TrainerName}");
            Console.WriteLine($"Metrics of best model from validation data --");
            PrintMulticlassClassificationMetrics(bestRun.ValidationMetrics);

            // STEP 4: Evaluate test data
            var testMetrics = mlContext.MulticlassClassification.CrossValidate(testingDataView, bestRun.Estimator, numberOfFolds: 5, labelColumnName: "reservation_status");
            Console.WriteLine($"Metrics of best model on test data --");
            PrintMulticlassClassificationFoldsAverageMetrics(testMetrics);
            
            // Save model
            SaveModel(mlContext, bestRun.Model, MODEL_FILEPATH, trainingDataView.Schema);
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            char separatorChar = Path.VolumeSeparatorChar == ':' ? '\\' : '/';
            relativePath = relativePath.Replace('\\', separatorChar);

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    MacroAccuracy = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    MicroAccuracy = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"    LogLoss for class {i + 1} = {metrics.PerClassLogLoss[i]:0.####}, the closer to 0, the better");
            }
            Console.WriteLine($"************************************************************");
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");

        }

        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }

    }
}
