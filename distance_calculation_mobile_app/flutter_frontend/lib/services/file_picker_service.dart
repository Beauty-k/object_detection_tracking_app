import 'package:file_picker/file_picker.dart';
import 'dart:io';

class FilePickerService {
  
  static Future<File?> pickVideoFile() async {
    try {

      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: false,
      );

      if (result != null && result.files.single.path != null) {
        // Convert picked file path to File object
        return File(result.files.single.path!);
      } else {
        return null;
      }
    } catch (e) {
      print("Error picking video file: $e");
      return null;
    }
  }
}
