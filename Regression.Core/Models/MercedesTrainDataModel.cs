using Microsoft.ML.Data;


namespace Regression.Core.Models
{
    /// <summary>
    /// Представляет модель строки обучающего набора данных для Mercedes
    /// </summary>
    public class MercedesTrainDataModel
    {
        [LoadColumn(0)]
        public uint Id { get; set; }

        [LoadColumn(1)]
        [ColumnName("Label")]
        public float Y { get; set; }

        [LoadColumn(2)]
        public string X0 { get; set; } = string.Empty;

        [LoadColumn(3)]
        public string X1 { get; set; } = string.Empty;

        [LoadColumn(4)]
        public string X2 { get; set; } = string.Empty;

        [LoadColumn(5)]
        public string X3 { get; set; } = string.Empty;

        [LoadColumn(6)]
        public string X4 { get; set; } = string.Empty;

        [LoadColumn(7)]
        public string X5 { get; set; } = string.Empty;

        [LoadColumn(8)]
        public string X6 { get; set; } = string.Empty;

        [LoadColumn(9)]
        public string X8 { get; set; } = string.Empty;

        [LoadColumn(10, 377)]
        [VectorType(368)]
        public float[] NumericColumns { get; set; } = [];
    }
}
