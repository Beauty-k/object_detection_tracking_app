import 'package:flutter/material.dart';
import '../helpers/snackbar_message.dart'; // assuming your showSnackBarMessage is here

Future<List<String>?> showObjectNameDialog(BuildContext context,
    {String? initialObj1, String? initialObj2}) async {
  final obj1Controller = TextEditingController(text: initialObj1);
  final obj2Controller = TextEditingController(text: initialObj2);

  return await showDialog<List<String>>(
    context: context,
    builder: (context) {
      return AlertDialog(
        title: const Text("Enter Object Names"),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: obj1Controller,
              decoration: const InputDecoration(labelText: "First object name"),
            ),
            TextField(
              controller: obj2Controller,
              decoration: const InputDecoration(labelText: "Second object name"),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, null),
            child: const Text("Cancel"),
          ),
          ElevatedButton(
            onPressed: () {
              final obj1 = obj1Controller.text.trim();
              final obj2 = obj2Controller.text.trim();
              if (obj1.isEmpty || obj2.isEmpty) {
                showSnackBarMessage(context, "Please enter both names", isError: true);
                return;
              }
              Navigator.pop(context, [obj1, obj2]);
            },
            child: const Text("OK"),
          ),
        ],
      );
    },
  );
}
