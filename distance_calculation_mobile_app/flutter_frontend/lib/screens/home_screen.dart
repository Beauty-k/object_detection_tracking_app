import 'dart:io';
import 'package:flutter/material.dart';
import '../reusable_widgets/custom_app_bar.dart';
import '../reusable_widgets/custom_button.dart';
import '../reusable_widgets/distance_card.dart';
import '../reusable_widgets/video_preview.dart';
import '../reusable_widgets/loading_overlay.dart';
import '../reusable_widgets/video_result_player.dart';
import '../services/file_picker_service.dart';
import '../services/api_service.dart';
import '../helpers/snackbar_message.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? selectedVideoFile;
  String apiResponse = "";
  bool isLoading = false;
  String? processedVideoUrl;
  double? distanceValue;

  Future<void> pickAndUploadVideo() async {
    final file = await FilePickerService.pickVideoFile();

    if (file == null) {
      showSnackBarMessage(context, "No video selected", isError: true);
      return;
    }

    setState(() {
      selectedVideoFile = file;
      apiResponse = "";
      processedVideoUrl = null;
      distanceValue = null;
      isLoading = true;
    });

    try {
      final response = await ApiService.uploadVideoAndCalculateDistance(
        videoFile: file,
        object1: "wallet",
        object2: "card",
      );

      // Extract distance safely
      double? backendDistance;
      if (response['distance'] != null) {
        backendDistance = double.tryParse(response['distance'].toString());
      }

      setState(() {
        distanceValue = backendDistance;
        apiResponse = backendDistance != null
            ? "Distance: ${backendDistance.toStringAsFixed(2)} mm"
            : "Distance not available";

        // If backend also returns processed video URL
        if (response['video_url'] != null) {
          processedVideoUrl = response['video_url'];
        }
      });

      showSnackBarMessage(context, "Video processed successfully!");
    } catch (e) {
      setState(() {
        apiResponse = "Upload failed: $e";
      });
      showSnackBarMessage(context, "Upload failed: $e", isError: true);
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return LoadingOverlay(
      isLoading: isLoading,
      child: Scaffold(
        appBar: const CustomAppBar(title: "Distance Measurement"),
        body: SingleChildScrollView(
          child: Column(
            children: [
              VideoPreview(
                videoPlayer: selectedVideoFile != null
                    ? Text("Selected Video:\n${selectedVideoFile!.path}")
                    : Container(
                        height: 200,
                        color: Colors.grey[300],
                        child: const Center(child: Text("No video selected")),
                      ),
              ),

              // Show actual backend distance
              DistanceCard(
                object1: "Wallet",
                object2: "Card",
                distance: distanceValue ?? 0.0,
              ),

              const SizedBox(height: 16),

              CustomButton(
                label: "Upload Video",
                onPressed: pickAndUploadVideo,
              ),

              const SizedBox(height: 16),

              Text(
                apiResponse,
                style: const TextStyle(fontSize: 16, color: Colors.green),
                textAlign: TextAlign.center,
              ),

              const SizedBox(height: 16),

              if (processedVideoUrl != null)
                VideoResultPlayer(videoUrl: processedVideoUrl!),
            ],
          ),
        ),
      ),
    );
  }
}
