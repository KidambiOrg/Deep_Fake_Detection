using DeepFakeDetection.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DeepFakeDetection.Business
{
    public class PredictionManager
    {
        private MLContext _mlContext;

        public PredictionEngine<ModelInput, ModelOutput> _predictionEngine;



        public PredictionManager()
        {
            // Initialize ML.NET context with a seed for reproducibility
            _mlContext = new MLContext(seed: 42);
        }


        private IOrderedEnumerable<KeyValuePair<string, float>> GetSortedScoresWithLabels(ModelOutput result)
        {
            var unlabeledScores = result.Score;
            var labelNames = GetLabels(result);

            Dictionary<string, float> labledScores = new Dictionary<string, float>();
            for (int i = 0; i < labelNames.Count(); i++)
            {
                // Map the names to the predicted result score array
                var labelName = labelNames.ElementAt(i);
                labledScores.Add(labelName.ToString(), unlabeledScores[i]);
            }

            return labledScores.OrderByDescending(c => c.Value);
        }

        /// <summary>
        /// Get the ordered label names.
        /// </summary>
        /// <param name="result">Predicted result to get the labels from.</param>
        /// <returns>List of labels.</returns>
        /// <exception cref="Exception"></exception>
        private IEnumerable<string> GetLabels(ModelOutput result)
        {
            var schema = _predictionEngine.OutputSchema;

            var labelColumn = schema.GetColumnOrNull("Label");
            if (labelColumn == null)
            {
                throw new Exception("Label column not found. Make sure the name searched for matches the name in the schema.");
            }

            // Key values contains an ordered array of the possible labels. This allows us to map the results to the correct label value.
            var keyNames = new VBuffer<ReadOnlyMemory<char>>();
            labelColumn.Value.GetKeyValues(ref keyNames);
            return keyNames.DenseValues().Select(x => x.ToString());
        }

        public void Predict(string imagePath, string modelPath = "")
        {
            // Load the trained model
            if (string.IsNullOrEmpty(modelPath))
            {
                modelPath = "DeepFakeImageModel_BestAccuracy.mlnet";
            }
            modelPath = "DeepFakeImageModel_BestAccuracy.mlnet";
            ITransformer trainedModel = _mlContext.Model.Load(modelPath, out var modelInputSchema);

            // Create a prediction engine
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            // Prepare the input data
            var imageBytes = File.ReadAllBytes(imagePath);
            var input = new ModelInput
            {
                ImageSource = imageBytes
            };


            // Make the prediction
            var prediction = _predictionEngine.Predict(input);
            var sortedScoresWithLabel = GetSortedScoresWithLabels(prediction);

            Console.WriteLine($"\nImage: {Path.GetFileName(imagePath)}");
            Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");

            Console.WriteLine($"Confidence Scores:");

            Console.WriteLine($"{"Class",-20}{"Score",-20}");
            Console.WriteLine($"{"-----",-20}{"-----",-20}");

            foreach (var score in sortedScoresWithLabel)
            {
                Console.WriteLine($"{score.Key,-20}{score.Value,-20}");
            }
        }
    }
}
