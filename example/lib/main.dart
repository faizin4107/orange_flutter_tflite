import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:orange_flutter_tflite/orange_flutter_tflite.dart';
import 'package:image_picker/image_picker.dart';

void main() => runApp(App());

class App extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: MyApp(),
    );
  }
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File? _image;
  // ignore: prefer_typing_uninitialized_variables
  var _recognitions;
  ImagePicker picker = ImagePicker();
  Future predictImagePicker() async {
    var image = await picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;
    await recognizeImage(image);
  }

  @override
  void initState() {
    super.initState();

    loadModel();
  }

  Future loadModel() async {
    OrangeFlutterTflite.close();
    try {
      await OrangeFlutterTflite.loadModel(
        model: "assets/model.tflite",
        labels: "assets/labels.txt",
      );
    } on PlatformException {
      debugPrint('Failed to load model.');
    }
  }

  

  Future recognizeImage(XFile image) async {
    int startTime = DateTime.now().millisecondsSinceEpoch;
    var recognitions = await OrangeFlutterTflite.runModelOnImage(
      path: image.path,
      numResults: 6,
      threshold: 0.05,
      imageMean: 127.5,
      imageStd: 127.5,
    );
    setState(() {
      _recognitions = recognitions;
    });
    int endTime = DateTime.now().millisecondsSinceEpoch;
    debugPrint("Inference took ${endTime - startTime}ms");
    debugPrint("recognitions $_recognitions");
  }

 

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(
        title: const Text('tflite example app'),
      ),
      body: Column(
        children: [
          _image == null
            ? const Center(child: Text('No image selected.'))
            : Container(
                decoration: BoxDecoration(
                    image: DecorationImage(
                        alignment: Alignment.topCenter,
                        image: MemoryImage(_recognitions),
                        fit: BoxFit.fill)),
                child: Opacity(opacity: 0.3, child: Image.file(_image!))),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: predictImagePicker,
        tooltip: 'Pick Image',
        child: const Icon(Icons.image),
      ),
    );
  }
}
