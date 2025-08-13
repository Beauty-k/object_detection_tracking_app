import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://10.0.2.2:8000/video";

  /// Uploads a video to the backend and calculates distance
  static Future<Map<String, dynamic>> uploadVideoAndCalculateDistance({
    required File videoFile,
    required String object1,
    required String object2,
  }) async {
    final uri = Uri.parse("$baseUrl/calculate-distance");

    final request = http.MultipartRequest('POST', uri);

    // Attach video file
    request.files.add(await http.MultipartFile.fromPath('file', videoFile.path));

    // Add form fields for object labels
    request.fields['label1'] = object1;
    request.fields['label2'] = object2;

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception("Failed to upload video: ${response.body}");
      }
    } catch (e) {
      throw Exception("Error uploading video: $e");
    }
  }
}

