using Microsoft.ML;
using Regression.CLI.Common;
using Regression.CLI.Data;
using Regression.Core.Models;
using Regression.Core.Training;

namespace Regression.CLI
{
    public class Program
    {
        private static readonly string _defaultTrainPath = Path.Combine(AppContext.BaseDirectory, "Data", "train_mercedes.csv");
        private static readonly string _defaultTestPath = Path.Combine(AppContext.BaseDirectory, "Data", "test_mercedes.csv");
        private static readonly string _defaultSubmissionPath = Path.Combine(AppContext.BaseDirectory, "submission.csv");

        private static void Main(string[] args)
        {
            var trainFilePath = args.Length > 0 ? args[0] : _defaultTrainPath;
            var testFilePath = args.Length > 1 ? args[1] : _defaultTestPath;

            var mlContext = new MLContext();

            // подгружаем данные из CSV файла в IDataView
            var loader = new MercedesDataFileService(mlContext);
            var trainRawData = loader.LoadTrainData(trainFilePath);

            ConsoleHelper.PrintDataPreview(trainRawData);

            // обучаем модель на обучающей выборке
            var modelTrainer = new MercedesModelTrainer(mlContext);

            var crossValidationResults = mlContext.Regression
                .CrossValidate( data: trainRawData, estimator: modelTrainer.CreatePipeline(), numberOfFolds: 5);

            ConsoleHelper.PrintCrossValidationMetrics("LightGBM", crossValidationResults);

            // после оценки обучаем итоговую модель на всех размеченных данных
            var finalModel = modelTrainer.Train(trainRawData);

            Console.WriteLine("\nИтоговая модель LightGBM обучена.\n");

            var testRawData = loader.LoadTestData(testFilePath);
            var testPredictions = finalModel.Transform(testRawData);

            var predictions = mlContext.Data
                .CreateEnumerable<MercedesPredictionModel>(testPredictions,reuseRowObject: false)
                .ToArray();

            var invalidPredictionCount = ConsoleHelper.PrintPredictionStatistics(predictions);

            if (invalidPredictionCount > 0)
            {
                throw new InvalidOperationException("Модель сформировала некорректные прогнозы.");
            }

            var submissionPath = args.Length > 2 ? args[2] : _defaultSubmissionPath;

            loader.WriteSubmission(submissionPath, predictions);

            Console.WriteLine($"\nРезультат сохранён: {submissionPath}");
        }
    }
}
