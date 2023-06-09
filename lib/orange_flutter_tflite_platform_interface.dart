import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'orange_flutter_tflite_method_channel.dart';

abstract class OrangeFlutterTflitePlatform extends PlatformInterface {
  /// Constructs a OrangeFlutterTflitePlatform.
  OrangeFlutterTflitePlatform() : super(token: _token);

  static final Object _token = Object();

  static OrangeFlutterTflitePlatform _instance = MethodChannelOrangeFlutterTflite();

  /// The default instance of [OrangeFlutterTflitePlatform] to use.
  ///
  /// Defaults to [MethodChannelOrangeFlutterTflite].
  static OrangeFlutterTflitePlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [OrangeFlutterTflitePlatform] when
  /// they register themselves.
  static set instance(OrangeFlutterTflitePlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
