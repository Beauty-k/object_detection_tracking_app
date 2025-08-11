import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

// import 'package:path/path.dart';
// import 'package:mime/mime.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? selectedFile;
  String? selectedFileName;
  String apiResponse = "";

  // File picker logic
  Future<void> pickVideo() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.video);
    if (result != null && result.files.single.path != null) {
      setState(() {
        selectedFile = File(result.files.single.path!);
        selectedFileName = result.files.single.name;
      });
    }
  }

  // Upload logic
  Future<void> uploadVideo() async {
    if (selectedFile == null) {
      setState(() {
        apiResponse = "Please choose a video first.";
      });
      return;
    }

    final uri = Uri.parse(
      "http://192.168.141.138:8000/video/calculate-distance",
    );

    final request = http.MultipartRequest('POST', uri);

    // Attach video file
    request.files.add(
      await http.MultipartFile.fromPath(
        'file', // key must match FastAPI param
        selectedFile!.path,
        contentType: MediaType('video', 'mp4'),
      ),
    );

    // Add labels
    request.fields['label1'] = 'blessing_card';
    request.fields['label2'] = 'wallet';

    // Send request
    final response = await request.send();

    // Handle response
    if (response.statusCode == 200) {
      final responseBody = await response.stream.bytesToString();
      setState(() {
        apiResponse = "Success: $responseBody";
      });
    } else {
      setState(() {
        apiResponse = "Failed: ${response.statusCode}";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Distance Measurement")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              "Upload a video to measure distance",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),

            ElevatedButton(
              onPressed: pickVideo,
              child: const Text("Choose Video"),
            ),
            const SizedBox(height: 10),

            Text(
              selectedFileName ?? "No file selected",
              style: const TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 20),

            ElevatedButton(
              onPressed: uploadVideo,
              child: const Text("Upload Video"),
            ),
            const SizedBox(height: 20),

            Text(apiResponse, style: const TextStyle(color: Colors.green)),
          ],
        ),
      ),
    );
  }
}
