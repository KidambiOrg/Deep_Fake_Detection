using DeepFakeDetection.Business;

namespace DeepFakeDetection.App
{
    internal class Program
    {

        private static string MLNetModelPath = Path.GetFullPath("DeepFakeImageModel.mlnet");


        static void Main(string[] args)
        {



            //// Initialize ML.NET context with a seed for reproducibility
            //var mlContext = new MLContext(seed: 42);

            //// Example 1: Basic Training (original method)
            //// TrainBasicModel(mlContext);

            //// Example 2: Enhanced Training (recommended for better accuracy)
            //// TrainEnhancedModel(mlContext);

            //// Example 3: Custom Training with specific architecture
            //// TrainCustomModel(mlContext);

            ////// Example 4: Full Training with Evaluation
            ////TrainAndEvaluateModel(mlContext);

            //TestModelPrediction(mlContext,
            //                    mlContext.Model.Load(MLNetModelPath, out var _),
            //                    @"C:\Kidambi\WorkRelated\DeepFake\sample_images\fake_4.png");


            new PredictionManager().Predict(@"C:\Kidambi\WorkRelated\DeepFake\sample_images\real_2.jpg");

            //new TrainingManager().Train(@"C:\Kidambi\WorkRelated\DeepFake\sample_images\archive1\train", "");

        }

        ///// <summary>
        ///// Example 1: Train using the basic pipeline (original auto-generated method)
        ///// </summary>
        //private static void TrainBasicModel(MLContext mlContext)
        //{
        //    Console.WriteLine("Training basic model...\n");

        //    // Load training data from folder structure
        //    // Expected structure:
        //    // - TrainingData/
        //    //   - REAL/
        //    //     - image1.jpg
        //    //     - image2.jpg
        //    //   - FAKE/
        //    //     - image1.jpg
        //    //     - image2.jpg
        //    string trainingDataPath = @"C:\Kidambi\WorkRelated\DeepFake\sample_images\archive1\train";
        //    var trainData = DeepFakeImageModel.LoadImageFromFolder(mlContext, trainingDataPath);

        //    // Train the model
        //    var model = DeepFakeImageModel.RetrainModel(mlContext, trainData);

        //    // Save the model
        //    string modelPath = "DeepFakeImageModel_Basic.mlnet";
        //    mlContext.Model.Save(model, trainData.Schema, modelPath);
        //    Console.WriteLine($"Model saved to: {modelPath}");
        //}

        ///// <summary>
        ///// Example 2: Train using the enhanced pipeline for better accuracy
        ///// </summary>
        //private static void TrainEnhancedModel(MLContext mlContext)
        //{
        //    Console.WriteLine("Training enhanced model with advanced options...\n");

        //    string trainingDataPath = @"C:\Kidambi\WorkRelated\DeepFake\sample_images\archive1\train";
        //    var trainData = DeepFakeImageModel.LoadImageFromFolder(mlContext, trainingDataPath);

        //    // Use enhanced training method
        //    var model = DeepFakeImageModel.RetrainEnhancedModel(mlContext, trainData);

        //    // Save the enhanced model
        //    string modelPath = "DeepFakeImageModel_Enhanced.mlnet";
        //    mlContext.Model.Save(model, trainData.Schema, modelPath);
        //    Console.WriteLine($"\nEnhanced model saved to: {modelPath}");
        //}

        ///// <summary>
        ///// Example 3: Train with custom architecture and hyperparameters
        ///// </summary>
        //private static void TrainCustomModel(MLContext mlContext)
        //{
        //    Console.WriteLine("Training custom model...\n");

        //    string trainingDataPath = @"C:\Kidambi\WorkRelated\DeepFake\sample_images\archive1\train";
        //    var trainData = DeepFakeImageModel.LoadImageFromFolder(mlContext, trainingDataPath);

        //    // Experiment with different architectures
        //    Console.WriteLine("Testing different architectures:\n");

        //    // Option 1: ResNetV2101 (Best for accuracy, slower)
        //    Console.WriteLine("1. Training with ResNetV2101 (High Accuracy)...");
        //    var resnetPipeline = DeepFakeImageModel.BuildCustomPipeline(
        //        mlContext,
        //        architecture: ImageClassificationTrainer.Architecture.ResnetV2101,
        //        epochs: 100,
        //        batchSize: 10,
        //        learningRate: 0.001f
        //    );
        //    var resnetModel = resnetPipeline.Fit(trainData);
        //    mlContext.Model.Save(resnetModel, trainData.Schema, "DeepFakeImageModel_ResNet101.mlnet");

        //    // Option 2: InceptionV3 (Balanced accuracy and speed)
        //    Console.WriteLine("\n2. Training with InceptionV3 (Balanced)...");
        //    var inceptionPipeline = DeepFakeImageModel.BuildCustomPipeline(
        //        mlContext,
        //        architecture: ImageClassificationTrainer.Architecture.InceptionV3,
        //        epochs: 100,
        //        batchSize: 16,
        //        learningRate: 0.001f
        //    );
        //    var inceptionModel = inceptionPipeline.Fit(trainData);
        //    mlContext.Model.Save(inceptionModel, trainData.Schema, "DeepFakeImageModel_Inception.mlnet");

        //    // Option 3: MobilenetV2 (Fastest, good for production)
        //    Console.WriteLine("\n3. Training with MobilenetV2 (Fast)...");
        //    var mobilenetPipeline = DeepFakeImageModel.BuildCustomPipeline(
        //        mlContext,
        //        architecture: ImageClassificationTrainer.Architecture.MobilenetV2,
        //        epochs: 100,
        //        batchSize: 32,
        //        learningRate: 0.01f
        //    );
        //    var mobilenetModel = mobilenetPipeline.Fit(trainData);
        //    mlContext.Model.Save(mobilenetModel, trainData.Schema, "DeepFakeImageModel_MobileNet.mlnet");

        //    Console.WriteLine("\nAll models trained and saved!");
        //}

        ///// <summary>
        ///// Example 4: Full training with proper train/test split and evaluation
        ///// </summary>
        //private static void TrainAndEvaluateModel(MLContext mlContext)
        //{
        //    Console.WriteLine("Training and evaluating model with train/test split...\n");

        //    // Load training data
        //    string trainingDataPath = @"C:\Kidambi\WorkRelated\DeepFake\sample_images\archive1\train";
        //    string testDataPath = @"C:\Kidambi\WorkRelated\DeepFake\sample_images\archive1\test";

        //    Console.WriteLine("Loading training data...");
        //    var trainData = DeepFakeImageModel.LoadImageFromFolder(mlContext, trainingDataPath);

        //    Console.WriteLine("Loading test data...");
        //    var testData = DeepFakeImageModel.LoadImageFromFolder(mlContext, testDataPath);

        //    // Train and evaluate
        //    //var (model, metrics) = DeepFakeImageModel.TrainAndEvaluate(mlContext, trainData, testData);
        //    var model = DeepFakeImageModel.TrainAndEvaluate(mlContext, trainData, testData, true);

        //    // Save the best model
        //    string modelPath = "DeepFakeImageModel_BestAccuracy.mlnet";
        //    mlContext.Model.Save(model, trainData.Schema, modelPath);
        //    Console.WriteLine($"\nModel saved to: {modelPath}");

        //    //// Test the model on a sample image
        //    //Console.WriteLine("\n=== Testing Model on Sample Image ===");
        //    //TestModelPrediction(mlContext, model, @"C:\Kidambi\WorkRelated\DeepFake\sample_images\archive1\test\FAKE\0 (10).jpg");
        //}

        ///// <summary>
        ///// Test model prediction on a single image
        ///// </summary>
        //private static void TestModelPrediction(MLContext mlContext, ITransformer model, string imagePath)
        //{
        //    var predictionEngine = mlContext.Model.CreatePredictionEngine<DeepFakeImageModel.ModelInput, DeepFakeImageModel.ModelOutput>(model);

        //    var imageBytes = File.ReadAllBytes(imagePath);
        //    var testInput = new DeepFakeImageModel.ModelInput
        //    {
        //        ImageSource = imageBytes
        //    };

        //    var prediction = predictionEngine.Predict(testInput);

        //    Console.WriteLine($"\nImage: {Path.GetFileName(imagePath)}");
        //    Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
        //    Console.WriteLine($"Confidence Scores:");

        //    // Get labels for the scores
        //    // Display scores using the labels from the model
        //    var labels = new[] { "FAKE", "REAL" }; // Adjust based on your folder structure
        //    for (int i = 0; i < prediction.Score.Length && i < labels.Length; i++)
        //    {
        //        Console.WriteLine($"  {labels[i],-10}: {prediction.Score[i]:P2}");
        //    }
        //}

        ///// <summary>
        ///// Helper method to split data into train and validation sets
        ///// </summary>
        //private static (IDataView TrainSet, IDataView ValidationSet) SplitTrainValidation(
        //    MLContext mlContext,
        //    IDataView data,
        //    double testFraction = 0.2)
        //{
        //    var split = mlContext.Data.TrainTestSplit(data, testFraction: testFraction, seed: 42);
        //    return (split.TrainSet, split.TestSet);
        //}





    }
}
