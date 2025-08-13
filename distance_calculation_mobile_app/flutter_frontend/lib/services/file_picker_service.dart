import 'package:file_picker/file_picker.dart';
import 'dart:io';

class FilePickerService {
  
  static Future<File?> pickVideoFile() async {
    try {
      // Opens file picker dialog and restricts it to video files
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: false,
      );

      if (result != null && result.files.single.path != null) {
        // Convert picked file path to File object
        return File(result.files.single.path!);
      } else {
        // User cancelled file picking
        return null;
      }
    } catch (e) {
      // Log or handle errors here
      print("Error picking video file: $e");
      return null;
    }
  }
}
