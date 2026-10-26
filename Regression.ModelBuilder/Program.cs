using Microsoft.ML;
using Regression_ModelBuilder;

internal class Program
{
    private static readonly string _defaultTrainPath =
        Path.Combine(AppContext.BaseDirectory, "Data", "train_mercedes.csv");

    private static void Main(string[] args)
    {
        var trainFilePath = args.Length > 0 ? args[0] : _defaultTrainPath;

        if (!File.Exists(trainFilePath))
        {
            throw new FileNotFoundException("Обучающий файл не найден.", trainFilePath);
        }

        var mlContext = new MLContext();

        var trainData = mlContext.Data
            .LoadFromTextFile<MercedesRegression.ModelInput>(trainFilePath, 
                                                             separatorChar: ',', 
                                                             hasHeader: true, 
                                                             allowQuoting: true, 
                                                             trimWhitespace: true);

        var modelBuilderPipeline = MercedesRegression.BuildPipeline(mlContext);

        var crossValidationResults = mlContext.Regression
            .CrossValidate(data: trainData,
                           estimator: modelBuilderPipeline,
                           numberOfFolds: 5,
                           labelColumnName: "y");

        var metrics = crossValidationResults
            .Select(result => result.Metrics)
            .ToArray();

        Console.WriteLine("\nСредние метрики pipeline из Model Builder");
        Console.WriteLine("-----------------------------------------");
        Console.WriteLine(
            $"Средняя абсолютная ошибка (MAE): " +
            $"{metrics.Average(x => x.MeanAbsoluteError):0.###}");

        Console.WriteLine(
            $"Средняя квадратичная ошибка (MSE): " +
            $"{metrics.Average(x => x.MeanSquaredError):0.###}");

        Console.WriteLine(
            $"Корень средней квадратичной ошибки (RMSE): " +
            $"{metrics.Average(x => x.RootMeanSquaredError):0.###}");

        Console.WriteLine(
            $"Коэффициент детерминации (R^2): " +
            $"{metrics.Average(x => x.RSquared):0.###}");
    }
}