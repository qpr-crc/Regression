using Microsoft.ML;
using Microsoft.ML.Transforms;
using Regression.Core.Models;

namespace Regression.Core.Preprocessing
{
    /// <summary>
    /// Преобразует исходные признаки набора данных Mercedes в числовой вектор Features
    /// </summary>
    public sealed class MercedesDataPreprocessor
    {
        private static readonly string[] CategoricalColumns =
        [
            nameof(MercedesTrainDataModel.X0),
        nameof(MercedesTrainDataModel.X1),
        nameof(MercedesTrainDataModel.X2),
        nameof(MercedesTrainDataModel.X3),
        nameof(MercedesTrainDataModel.X4),
        nameof(MercedesTrainDataModel.X5),
        nameof(MercedesTrainDataModel.X6),
        nameof(MercedesTrainDataModel.X8)
        ];

        private readonly MLContext _mlContext;

        public MercedesDataPreprocessor(
            MLContext mlContext)
        {
            _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        }

        /// <summary>
        /// Создаёт pipeline подготовки признаков.
        /// </summary>
        public IEstimator<ITransformer> CreatePipeline()
        {
            var encodedColumns = CategoricalColumns
                .Select(column =>
                    new InputOutputColumnPair(outputColumnName: $"{column}Encoded",
                        inputColumnName: column))
                .ToArray();

            var featureColumns = encodedColumns
                .Select(column => column.OutputColumnName)
                .Append(nameof(MercedesTrainDataModel.NumericColumns))
                .ToArray();

            return _mlContext.Transforms.Categorical
                .OneHotEncoding(
                    columns: encodedColumns,
                    outputKind: OneHotEncodingEstimator.OutputKind.Indicator)
                .Append(_mlContext.Transforms.Concatenate("Features", featureColumns));
        }
    }
}
