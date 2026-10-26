using Microsoft.ML;
using Microsoft.ML.Data;
using Regression.Core.Models;
using static Microsoft.ML.TrainCatalogBase;

namespace Regression.CLI.Common
{
    /// <summary>
    /// Выводит данные, метрики и результаты работы модели в консоль.
    /// </summary>
    public static class ConsoleHelper
    {
        /// <summary>
        /// Выводит средние метрики регрессии по всем фолдам.
        /// </summary>
        public static void PrintCrossValidationMetrics(
            string algorithmName,
            IReadOnlyList<CrossValidationResult<RegressionMetrics>> results)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(algorithmName);
            ArgumentNullException.ThrowIfNull(results);

            if (results.Count == 0)
            {
                throw new ArgumentException("Список результатов кросс-валидации пуст.", nameof(results));
            }

            var metrics = results.Select(result => result.Metrics).ToArray();

            PrintHeader(
                $"Средние метрики кросс-валидации модели {algorithmName}");

            Console.WriteLine(
                $"Средняя абсолютная ошибка (MAE): " +
                $"{metrics.Average(metric => metric.MeanAbsoluteError):0.###}");

            Console.WriteLine(
                $"Средняя квадратичная ошибка (MSE): " +
                $"{metrics.Average(metric => metric.MeanSquaredError):0.###}");

            Console.WriteLine(
                $"Корень средней квадратичной ошибки (RMSE): " +
                $"{metrics.Average(metric => metric.RootMeanSquaredError):0.###}");

            Console.WriteLine(
                $"Коэффициент детерминации (R^2): " +
                $"{metrics.Average(metric => metric.RSquared):0.###}");
        }

        /// <summary>
        /// Показывает несколько первых строк IDataView.
        /// </summary>
        public static void PrintDataPreview(
            IDataView dataView,
            int numberOfRows = 4)
        {
            ArgumentNullException.ThrowIfNull(dataView);

            PrintHeader(
                $"Первые {numberOfRows} строки загруженных данных");

            var preview = dataView.Preview(maxRows: numberOfRows);

            foreach (var row in preview.RowView)
            {
                var values = row.Values.Select(
                    column => $"{column.Key}: {column.Value}");

                Console.WriteLine(
                    $"Строка: {string.Join(" | ", values)}\n");
            }
        }

        /// <summary>
        /// Выводит общую статистику прогнозов.
        /// </summary>
        /// <returns>Количество некорректных прогнозов.</returns>
        public static int PrintPredictionStatistics(
            IReadOnlyCollection<MercedesPredictionModel> predictions)
        {
            ArgumentNullException.ThrowIfNull(predictions);

            var validPredictions = predictions
                .Where(prediction => float.IsFinite(prediction.PredictedY))
                .ToArray();

            var invalidPredictionCount =
                predictions.Count - validPredictions.Length;

            PrintHeader("Статистика прогнозов");

            Console.WriteLine($"Всего прогнозов: {predictions.Count}");
            Console.WriteLine(
                $"Некорректных прогнозов: {invalidPredictionCount}");

            if (validPredictions.Length > 0)
            {
                Console.WriteLine(
                    $"Минимальный прогноз: " +
                    $"{validPredictions.Min(x => x.PredictedY):F4}");

                Console.WriteLine(
                    $"Максимальный прогноз: " +
                    $"{validPredictions.Max(x => x.PredictedY):F4}");

                Console.WriteLine(
                    $"Средний прогноз: " +
                    $"{validPredictions.Average(x => x.PredictedY):F4}");
            }

            return invalidPredictionCount;
        }

        /// <summary>
        /// Выводит заголовок консольного раздела.
        /// </summary>
        private static void PrintHeader(string title)
        {
            var defaultColor = Console.ForegroundColor;

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"\n{title}");
            Console.WriteLine(new string('-', title.Length));
            Console.ForegroundColor = defaultColor;
        }
    }
}
