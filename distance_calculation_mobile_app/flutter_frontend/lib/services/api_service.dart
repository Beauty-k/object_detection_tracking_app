    // import 'dart:convert';
// import 'package:http/http.dart' as http;
// import 'package:http_parser/http_parser.dart';

// class ApiService {
//   // Replace with your machine’s IP address running the FastAPI server
//   static const String baseUrl = 'http://192.168.141.138:8000';

//   static Future<String> sendVideoToBackend(String filePath, String label1, String label2) async {
//     var uri = Uri.parse('$baseUrl/video/calculate-distance');
//     var request = http.MultipartRequest('POST', uri);

//     request.files.add(await http.MultipartFile.fromPath(
//       'file',
//       filePath,
//       contentType: MediaType('video', 'mp4'),
//     ));

//     request.fields['label1'] = label1;
//     request.fields['label2'] = label2;

//     var response = await request.send();

//     if (response.statusCode == 200) {
//       final res = await http.Response.fromStream(response);
//       final decoded = jsonDecode(res.body);

//       // Make sure the backend returns 'output_video_path' key
//       return "$baseUrl/${decoded['output_video_path']}";
//     } else {
//       throw Exception('Failed to process video. Status code: ${response.statusCode}');
//     }
//   }
// }
