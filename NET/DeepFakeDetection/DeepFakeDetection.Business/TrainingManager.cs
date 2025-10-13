using DeepFakeDetection.Models;
using Microsoft.ML;
using Microsoft.ML.Vision;
using System.Collections.Concurrent;

namespace DeepFakeDetection.Business
{
    public class TrainingManager
    {


        private MLContext _mlContext;

        public TrainingManager()
        {
            _mlContext = new MLContext(seed: 42);
        }


        public void Train(string trainDataPath, string testDataPath, bool enableEarlyStopping = true, bool evaluateModel = false, string modelPath = "")
        {
            IDataView trainData = LoadImageFromFolder(_mlContext, trainDataPath);

            //Train the model.
            ITransformer model = TrainAndEvaluate(_mlContext, trainData, testDataPath, enableEarlyStopping, evaluateModel);

            // Save the model
            if (string.IsNullOrEmpty(modelPath))
            {
                modelPath = "DeepFakeImageModel_BestAccuracy.mlnet";
            }

            _mlContext.Model.Save(model, trainData.Schema, modelPath);

        }


        /// <summary>
        ///  Load an IDataView from a folder path.
        /// </summary>
        /// <param name="mlContext">The common context for all ML.NET operations.</param>
        /// <param name="folder"> Folder to the image data for training.</param>
        private IDataView LoadImageFromFolder(MLContext mlContext, string folder)
        {
            var res = new List<ModelInput>();
            var allowedImageExtensions = new[] { ".png", ".jpg", ".jpeg", ".gif", "webp" };
            DirectoryInfo rootDirectoryInfo = new DirectoryInfo(folder);
            DirectoryInfo[] subDirectories = rootDirectoryInfo.GetDirectories();

            if (subDirectories.Length == 0)
            {
                throw new Exception("fail to find subdirectories");
            }

            var concurrentRes = new ConcurrentBag<ModelInput>();

            Console.WriteLine($"Found {subDirectories.Length} subdirectories");

            foreach (var directory in subDirectories)
            {
                // Remove .Take(10) to load ALL images
                var imageList = directory.EnumerateFiles()
                    .Where(f => allowedImageExtensions.Contains(f.Extension.ToLower()))
                    .ToList(); // Convert to list to avoid enumeration issues

                Console.WriteLine($"Directory: {directory.Name} - Found {imageList.Count} images");

                if (imageList.Any())
                {
                    Parallel.ForEach(imageList, image =>
                    {
                        try
                        {
                            var imageBytes = File.ReadAllBytes(image.FullName);
                            concurrentRes.Add(new ModelInput
                            {
                                Label = directory.Name,
                                ImageSource = imageBytes,
                            });
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error loading image {image.Name}: {ex.Message}");
                        }
                    });
                }
            }

            var totalImages = concurrentRes.Count;
            Console.WriteLine($"Total images loaded: {totalImages}");

            if (totalImages == 0)
            {
                throw new Exception("No images were loaded. Please check the directory structure and file extensions.");
            }

            return mlContext.Data.LoadFromEnumerable(concurrentRes.AsEnumerable());
        }

        /// <summary>
        /// Retrain model using the pipeline generated as part of the training process.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainData"></param>
        /// <returns></returns>
        private ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        private IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Label", inputColumnName: @"Label", addKeyValueAnnotationsAsText: false)
                                    .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName: @"Label", scoreColumnName: @"Score", featureColumnName: @"ImageSource"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

            return pipeline;
        }

        /// <summary>
        /// Build an enhanced pipeline with advanced options for better accuracy.
        /// This includes better architecture and optimized hyperparameters.
        /// </summary>
        /// <param name="mlContext">The ML.NET context</param>
        /// <param name="enableEarlyStopping">Whether to enable early stopping (requires TestOnTrainSet)</param>
        /// <returns>Enhanced training pipeline</returns>
        private IEstimator<ITransformer> BuildEnhancedPipeline(MLContext mlContext, bool enableEarlyStopping = false)
        {
            // Configure advanced image classification options for better accuracy
            var options = new ImageClassificationTrainer.Options
            {
                // Feature extraction settings
                FeatureColumnName = "ImageSource",
                LabelColumnName = "Label",

                //// Architecture: Use more sophisticated models for better accuracy
                //// Options: ResnetV2101, ResnetV250, InceptionV3, MobilenetV2
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101, // Deep architecture for high accuracy

                // Training hyperparameters - optimized for accuracy
                Epoch = 100,                    // More epochs for better convergence (default: 50)
                BatchSize =16,                 // Smaller batch size for better gradient estimates
                LearningRate = 0.0001f,          // Lower learning rate for fine-tuning

                // Performance optimization
                WorkspacePath = Path.Combine(Path.GetTempPath(), "MLNet_ImageClassification"),

                // TestOnTrainSet must be true for early stopping to work
                TestOnTrainSet = true,

                // Reuse pre-trained weights for transfer learning
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true,
            };

            // Only add early stopping if enabled (requires TestOnTrainSet = true)
            if (enableEarlyStopping)
            {
                options.EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping
                {
                    MinDelta = 0.001f,          // Minimum improvement threshold
                    Patience = 20               // Stop if no improvement after 20 epochs
                };
            }

            // Build enhanced pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                                outputColumnName: "Label",
                                inputColumnName: "Label",
                                addKeyValueAnnotationsAsText: false)

                            // Train with enhanced options
                            .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(options))

                            // Convert predicted label key back to original values
                            .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                                outputColumnName: "PredictedLabel",
                                inputColumnName: "PredictedLabel"));


            // Data process configuration with pipeline data transformations
            //var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Label", inputColumnName: @"Label", addKeyValueAnnotationsAsText: false)
            //                        .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName: @"Label", scoreColumnName: @"Score", featureColumnName: @"ImageSource"))
            //                        .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));



            return pipeline;
        }

        /// <summary>
        /// Build a custom pipeline with configurable parameters for experimentation.
        /// </summary>
        /// <param name="mlContext">The ML.NET context</param>
        /// <param name="architecture">Model architecture (ResnetV2101, InceptionV3, etc.)</param>
        /// <param name="epochs">Number of training epochs</param>
        /// <param name="batchSize">Training batch size</param>
        /// <param name="learningRate">Learning rate</param>
        /// <param name="enableEarlyStopping">Whether to enable early stopping (requires TestOnTrainSet)</param>
        /// <returns>Custom training pipeline</returns>
        private IEstimator<ITransformer> BuildCustomPipeline(
            MLContext mlContext,
            ImageClassificationTrainer.Architecture architecture = ImageClassificationTrainer.Architecture.ResnetV2101,
            int epochs = 100,
            int batchSize = 10,
            float learningRate = 0.0001f,
            bool enableEarlyStopping = false)
        {
            var options = new ImageClassificationTrainer.Options
            {
                FeatureColumnName = "ImageSource",
                LabelColumnName = "Label",
                ScoreColumnName = "Score",
                Arch = architecture,
                Epoch = epochs,
                BatchSize = batchSize,
                LearningRate = learningRate,
                TestOnTrainSet = true,  // Required for monitoring and early stopping
                WorkspacePath = Path.Combine(Path.GetTempPath(), "MLNet_ImageClassification"),
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true,
            };

            // Only add early stopping if enabled (requires TestOnTrainSet = true)
            if (enableEarlyStopping)
            {
                options.EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping
                {
                    MinDelta = 0.001f,
                    Patience = 15
                };
            }

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label", addKeyValueAnnotationsAsText: false)
                                    .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(options))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));




            return pipeline;
        }

        /// <summary>
        /// Retrain model using the enhanced pipeline for better accuracy.
        /// </summary>
        /// <param name="mlContext">The ML.NET context</param>
        /// <param name="trainData">Training data</param>
        /// <param name="validationData">Optional validation data for better training monitoring</param>
        /// <returns>Trained model</returns>
        private ITransformer RetrainEnhancedModel(
            MLContext mlContext,
            IDataView trainData,
            IDataView validationData = null)
        {
            Console.WriteLine("Training enhanced model with advanced options...");
            Console.WriteLine("This may take longer but should achieve better accuracy.");

            var pipeline = BuildEnhancedPipeline(mlContext);

            var model = pipeline.Fit(trainData);

            Console.WriteLine("Training completed!");

            return model;
        }

        /// <summary>
        /// Train and evaluate model with comprehensive metrics reporting.
        /// </summary>
        /// <param name="mlContext">The ML.NET context</param>
        /// <param name="trainData">Training data</param>
        /// <param name="testData">Test data for evaluation</param>
        /// <param name="enableEarlyStopping">Whether to enable early stopping during training</param>
        /// <returns>Trained model with evaluation metrics</returns>
        //public static (ITransformer Model, MulticlassClassificationMetrics Metrics) TrainAndEvaluate(
        private ITransformer TrainAndEvaluate(
        MLContext mlContext,
        IDataView trainData,
        string testDataPath,
        bool enableEarlyStopping = false,
        bool evaluateModel = false)
        {

            // Build pipeline with or without early stopping
            var pipeline = BuildEnhancedPipeline(mlContext, enableEarlyStopping: enableEarlyStopping);

            // Train the model
            var model = pipeline.Fit(trainData);

            Console.WriteLine("\nTraining completed!");
            //Console.WriteLine("Evaluating model on test set...");

            if (evaluateModel)
            {
                IDataView testData = LoadImageFromFolder(_mlContext, testDataPath);

                var predictions = model.Transform(testData);
                var metrics = mlContext.MulticlassClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score");
            }


            //// Print detailed metrics
            //Console.WriteLine("\n=== Model Evaluation Metrics ===");
            //Console.WriteLine($"MacroAccuracy:    {metrics.MacroAccuracy:P2}");
            //Console.WriteLine($"MicroAccuracy:    {metrics.MicroAccuracy:P2}");
            //Console.WriteLine($"LogLoss:          {metrics.LogLoss:F4}");
            //Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction:F4}");

            //// Per-class metrics
            //Console.WriteLine("\n=== Per-Class Metrics ===");
            //for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            //{
            //    Console.WriteLine($"Class {i}: LogLoss={metrics.PerClassLogLoss[i]:F4}");
            //}

            //// Confusion matrix
            //Console.WriteLine("\n=== Confusion Matrix ===");
            //Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

            //return (model, metrics);
            return model;
        }


    }
}
