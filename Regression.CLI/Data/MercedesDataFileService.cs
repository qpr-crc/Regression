using Microsoft.ML;
using Regression.Core.Models;
using System.Globalization;
using System.Text;

namespace Regression.CLI.Data
{
    /// <summary>
    /// Загружает обучающие и тестовые данные Mercedes
    /// </summary>
    public sealed class MercedesDataFileService
    {
        private readonly MLContext _mlContext;

        public MercedesDataFileService(MLContext mlContext)
        {
            _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        }

        /// <summary>
        /// Загружает обучающую выборку.
        /// </summary>
        /// <param name="dataPath">Путь к файлу данных</param>
        /// <returns>Входные и выходные данные операторов запросов</returns>
        public IDataView LoadTrainData(string dataPath)
        {
            return Load<MercedesTrainDataModel>(dataPath);
        }

        /// <summary>
        /// Загружает тестовую выборку.
        /// </summary>
        /// <param name="dataPath">Путь к файлу данных</param>
        /// <returns>Входные и выходные данные операторов запросов</returns>
        public IDataView LoadTestData(string dataPath)
        {
            return Load<MercedesTestDataModel>(dataPath);
        }

        /// <summary>
        /// Сохраняет прогнозы в CSV-файл.
        /// </summary>
        public void WriteSubmission(
            string filePath,
            IEnumerable<MercedesPredictionModel> predictions)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
            ArgumentNullException.ThrowIfNull(predictions);

            using var writer = new StreamWriter(
                filePath,
                append: false,
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            writer.WriteLine("ID,y");

            foreach (var prediction in predictions)
            {
                writer.Write(prediction.Id.ToString(CultureInfo.InvariantCulture));
                writer.Write(',');
                writer.WriteLine(prediction.PredictedY.ToString("R", CultureInfo.InvariantCulture));
            }
        }

        private IDataView Load<T>(string dataPath)
            where T : class
        {
            if (string.IsNullOrWhiteSpace(dataPath))
            {
                throw new ArgumentException("Путь к данным не может быть пустым.", nameof(dataPath));
            }

            if (!File.Exists(dataPath))
            {
                throw new FileNotFoundException("Файл данных не найден.", dataPath);
            }

            return _mlContext.Data.LoadFromTextFile<T>(
                dataPath,
                separatorChar: ',',
                hasHeader: true,
                allowQuoting: true,
                trimWhitespace: true);
        }
    }
}