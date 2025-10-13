using Microsoft.ML.Data;

namespace DeepFakeDetection.Models
{
    public class ModelOutput
    {
        [ColumnName(@"Label")]
        public uint Label { get; set; }

        [ColumnName(@"ImageSource")]
        public byte[] ImageSource { get; set; }

        [ColumnName(@"PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; }

    }
}
