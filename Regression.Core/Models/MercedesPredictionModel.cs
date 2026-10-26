using Microsoft.ML.Data;

namespace Regression.Core.Models
{
    /// <summary>
    /// Содержит идентификатор строки и прогноз LightGBM
    /// </summary>
    public class MercedesPredictionModel
    {
        public uint Id { get; set; }

        [ColumnName("Score")]
        public float PredictedY { get; set; }
    }
}
