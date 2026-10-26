using Microsoft.ML.Data;

namespace Regression.Core.Models
{
    /// <summary>
    /// Представляет строку тестового набора Mercedes
    /// </summary>
    public class MercedesTestDataModel
    {
        [LoadColumn(0)]
        public uint Id { get; set; }

        [LoadColumn(1)]
        public string X0 { get; set; } = string.Empty;

        [LoadColumn(2)]
        public string X1 { get; set; } = string.Empty;

        [LoadColumn(3)]
        public string X2 { get; set; } = string.Empty;

        [LoadColumn(4)]
        public string X3 { get; set; } = string.Empty;

        [LoadColumn(5)]
        public string X4 { get; set; } = string.Empty;

        [LoadColumn(6)]
        public string X5 { get; set; } = string.Empty;

        [LoadColumn(7)]
        public string X6 { get; set; } = string.Empty;

        [LoadColumn(8)]
        public string X8 { get; set; } = string.Empty;

        [LoadColumn(9, 376)]
        [VectorType(368)]
        public float[] NumericColumns { get; set; } = [];
    }
}
