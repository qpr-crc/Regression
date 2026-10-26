using Microsoft.ML;
using Regression.Core.Preprocessing;

namespace Regression.Core.Training
{
    /// <summary>
    /// Создаёт и обучает регрессионную модель LightGBM для набора данных Mercedes, используя подготовленные признаки
    /// </summary>
    public sealed class MercedesModelTrainer
    {
        private readonly MLContext _mlContext;
        private readonly MercedesDataPreprocessor _preprocessor;

        public MercedesModelTrainer(MLContext mlContext)
        {
            _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
            _preprocessor = new MercedesDataPreprocessor(mlContext);
        }

        /// <summary>
        /// Создаёт полный pipeline
        /// </summary>
        public IEstimator<ITransformer> CreatePipeline()
        {
            var lightGbm = _mlContext.Regression.Trainers.LightGbm();

            return _preprocessor
                .CreatePipeline()
                .Append(lightGbm);
        }

        /// <summary>
        /// Обучает полный pipeline на переданных данных
        /// </summary>
        public ITransformer Train(IDataView trainData)
        {
            ArgumentNullException.ThrowIfNull(trainData);

            return CreatePipeline()
                .Fit(trainData);
        }
    }
}
