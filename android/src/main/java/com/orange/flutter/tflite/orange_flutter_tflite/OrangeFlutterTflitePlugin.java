package com.orange.flutter.tflite.orange_flutter_tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;

import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.PluginRegistry.Registrar;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;


public class OrangeFlutterTflitePlugin implements MethodCallHandler {
  private final Registrar mRegistrar;
  private Interpreter tfLite;
  private boolean tfLiteBusy = false;
  private int inputSize = 0;
  private Vector<String> labels;
  float[][] labelProb;
  private static final int BYTES_PER_CHANNEL = 4;

  String[] partNames = {
      "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
      "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
      "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
  };

  String[][] poseChain = {
      {"nose", "leftEye"}, {"leftEye", "leftEar"}, {"nose", "rightEye"},
      {"rightEye", "rightEar"}, {"nose", "leftShoulder"},
      {"leftShoulder", "leftElbow"}, {"leftElbow", "leftWrist"},
      {"leftShoulder", "leftHip"}, {"leftHip", "leftKnee"},
      {"leftKnee", "leftAnkle"}, {"nose", "rightShoulder"},
      {"rightShoulder", "rightElbow"}, {"rightElbow", "rightWrist"},
      {"rightShoulder", "rightHip"}, {"rightHip", "rightKnee"},
      {"rightKnee", "rightAnkle"}
  };

  Map<String, Integer> partsIds = new HashMap<>();
  List<Integer> parentToChildEdges = new ArrayList<>();
  List<Integer> childToParentEdges = new ArrayList<>();

  public static void registerWith(Registrar registrar) {
    final MethodChannel channel = new MethodChannel(registrar.messenger(), "orange_flutter_tflite");
    channel.setMethodCallHandler(new OrangeFlutterTflitePlugin(registrar));
  }

  private OrangeFlutterTflitePlugin(Registrar registrar) {
    this.mRegistrar = registrar;
  }

  @Override
  public void onMethodCall(MethodCall call, Result result) {
    if (call.method.equals("loadModel")) {
      try {
        String res = loadModel((HashMap) call.arguments);
        result.success(res);
      } catch (Exception e) {
        result.error("Failed to load model", e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnImage")) {
      try {
        new RunModelOnImage((HashMap) call.arguments, result).executeTfliteTask();
      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("close")) {
      close();
    } else {
      result.error("Invalid method", call.method.toString(), "");
    }
  }

  private String loadModel(HashMap args) throws IOException {
    String model = args.get("model").toString();
    Object isAssetObj = args.get("isAsset");
    boolean isAsset = isAssetObj == null ? false : (boolean) isAssetObj;
    MappedByteBuffer buffer = null;
    String key = null;
    AssetManager assetManager = null;
    if (isAsset) {
      assetManager = mRegistrar.context().getAssets();
      key = mRegistrar.lookupKeyForAsset(model);
      AssetFileDescriptor fileDescriptor = assetManager.openFd(key);
      FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    } else {
      FileInputStream inputStream = new FileInputStream(new File(model));
      FileChannel fileChannel = inputStream.getChannel();
      long declaredLength = fileChannel.size();
      buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, declaredLength);
    }

    int numThreads = (int) args.get("numThreads");
    Boolean useGpuDelegate = (Boolean) args.get("useGpuDelegate");
    if (useGpuDelegate == null) {
      useGpuDelegate = false;
    }

    final Interpreter.Options tfliteOptions = new Interpreter.Options();
    tfliteOptions.setNumThreads(numThreads);
    if (useGpuDelegate){
      GpuDelegate delegate = new GpuDelegate();
      tfliteOptions.addDelegate(delegate);
    }
    tfLite = new Interpreter(buffer, tfliteOptions);

    String labels = args.get("labels").toString();

    if (labels.length() > 0) {
      if (isAsset) {
        key = mRegistrar.lookupKeyForAsset(labels);
        loadLabels(assetManager, key);
      } else {
        loadLabels(null, labels);
      }
    }

    return "success";
  }

  private void loadLabels(AssetManager assetManager, String path) {
    BufferedReader br;
    try {
      if (assetManager != null) {
        br = new BufferedReader(new InputStreamReader(assetManager.open(path)));
      } else {
        br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(path))));
      }
      String line;
      labels = new Vector<>();
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      labelProb = new float[1][labels.size()];
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Failed to read label file", e);
    }
  }

  private List<Map<String, Object>> GetTopN(int numResults, float threshold) {
    PriorityQueue<Map<String, Object>> pq =
        new PriorityQueue<>(
            1,
            new Comparator<Map<String, Object>>() {
              @Override
              public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                return Float.compare((float) rhs.get("confidence"), (float) lhs.get("confidence"));
              }
            });

    for (int i = 0; i < labels.size(); ++i) {
      float confidence = labelProb[0][i];
      if (confidence > threshold) {
        Map<String, Object> res = new HashMap<>();
        res.put("index", i);
        res.put("label", labels.size() > i ? labels.get(i) : "unknown");
        res.put("confidence", confidence);
        pq.add(res);
      }
    }

    final ArrayList<Map<String, Object>> recognitions = new ArrayList<>();
    int recognitionsSize = Math.min(pq.size(), numResults);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }

    return recognitions;
  }

  Bitmap feedOutput(ByteBuffer imgData, float mean, float std) {
    Tensor tensor = tfLite.getOutputTensor(0);
    int outputSize = tensor.shape()[1];
    Bitmap bitmapRaw = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888);

    if (tensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
          int pixelValue = 0xFF << 24;
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF) << 16);
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF) << 8);
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF));
          bitmapRaw.setPixel(j, i, pixelValue);
        }
      }
    } else {
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
          int pixelValue = 0xFF << 24;
          pixelValue |= ((imgData.get() & 0xFF) << 16);
          pixelValue |= ((imgData.get() & 0xFF) << 8);
          pixelValue |= ((imgData.get() & 0xFF));
          bitmapRaw.setPixel(j, i, pixelValue);
        }
      }
    }
    return bitmapRaw;
  }

  ByteBuffer feedInputTensor(Bitmap bitmapRaw, float mean, float std) throws IOException {
    Tensor tensor = tfLite.getInputTensor(0);
    int[] shape = tensor.shape();
    inputSize = shape[1];
    int inputChannels = shape[3];

    int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
    ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
    imgData.order(ByteOrder.nativeOrder());

    Bitmap bitmap = bitmapRaw;
    if (bitmapRaw.getWidth() != inputSize || bitmapRaw.getHeight() != inputSize) {
      Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
          inputSize, inputSize, false);
      bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
      final Canvas canvas = new Canvas(bitmap);
      if (inputChannels == 1){
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        canvas.drawBitmap(bitmapRaw, matrix, paint);
      } else {
        canvas.drawBitmap(bitmapRaw, matrix, null);
      }
    }

    if (tensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          int pixelValue = bitmap.getPixel(j, i);
          if (inputChannels > 1){
            imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
            imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
            imgData.putFloat(((pixelValue & 0xFF) - mean) / std);
          } else {
            imgData.putFloat((((pixelValue >> 16 | pixelValue >> 8 | pixelValue) & 0xFF) - mean) / std);
          }
        }
      }
    } else {
      for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
          int pixelValue = bitmap.getPixel(j, i);
          if (inputChannels > 1){
            imgData.put((byte) ((pixelValue >> 16) & 0xFF));
            imgData.put((byte) ((pixelValue >> 8) & 0xFF));
            imgData.put((byte) (pixelValue & 0xFF));
          } else {
            imgData.put((byte) ((pixelValue >> 16 | pixelValue >> 8 | pixelValue) & 0xFF));
          }
        }
      }
    }

    return imgData;
  }

  ByteBuffer feedInputTensorImage(String path, float mean, float std) throws IOException {
    InputStream inputStream = new FileInputStream(path.replace("file://", ""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);

    return feedInputTensor(bitmapRaw, mean, std);
  }

  ByteBuffer feedInputTensorFrame(List<byte[]> bytesList, int imageHeight, int imageWidth, float mean, float std, int rotation) throws IOException {
    ByteBuffer Y = ByteBuffer.wrap(bytesList.get(0));
    ByteBuffer U = ByteBuffer.wrap(bytesList.get(1));
    ByteBuffer V = ByteBuffer.wrap(bytesList.get(2));

    int Yb = Y.remaining();
    int Ub = U.remaining();
    int Vb = V.remaining();

    byte[] data = new byte[Yb + Ub + Vb];

    Y.get(data, 0, Yb);
    V.get(data, Yb, Vb);
    U.get(data, Yb + Vb, Ub);

    Bitmap bitmapRaw = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
    Allocation bmData = renderScriptNV21ToRGBA888(
        mRegistrar.context(),
        imageWidth,
        imageHeight,
        data);
    bmData.copyTo(bitmapRaw);

    Matrix matrix = new Matrix();
    matrix.postRotate(rotation);
    bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);

    return feedInputTensor(bitmapRaw, mean, std);
  }

  public Allocation renderScriptNV21ToRGBA888(Context context, int width, int height, byte[] nv21) {
    // https://stackoverflow.com/a/36409748
    RenderScript rs = RenderScript.create(context);
    ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

    Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
    Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

    Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
    Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

    in.copyFrom(nv21);

    yuvToRgbIntrinsic.setInput(in);
    yuvToRgbIntrinsic.forEach(out);
    return out;
  }

  private abstract class TfliteTask extends AsyncTask<Void, Void, Void> {
    Result result;
    boolean asynch;

    TfliteTask(HashMap args, Result result) {
      if (tfLiteBusy) throw new RuntimeException("Interpreter busy");
      else tfLiteBusy = true;
      Object asynch = args.get("asynch");
      this.asynch = asynch == null ? false : (boolean) asynch;
      this.result = result;
    }

    abstract void runTflite();

    abstract void onRunTfliteDone();

    public void executeTfliteTask() {
      if (asynch) execute();
      else {
        runTflite();
        tfLiteBusy = false;
        onRunTfliteDone();
      }
    }

    protected Void doInBackground(Void... backgroundArguments) {
      runTflite();
      return null;
    }

    protected void onPostExecute(Void backgroundResult) {
      tfLiteBusy = false;
      onRunTfliteDone();
    }
  }

  private class RunModelOnImage extends TfliteTask {
    int NUM_RESULTS;
    float THRESHOLD;
    ByteBuffer input;
    long startTime;

    RunModelOnImage(HashMap args, Result result) throws IOException {
      super(args, result);

      String path = args.get("path").toString();
      double mean = (double) (args.get("imageMean"));
      float IMAGE_MEAN = (float) mean;
      double std = (double) (args.get("imageStd"));
      float IMAGE_STD = (float) std;
      NUM_RESULTS = (int) args.get("numResults");
      double threshold = (double) args.get("threshold");
      THRESHOLD = (float) threshold;

      startTime = SystemClock.uptimeMillis();
      input = feedInputTensorImage(path, IMAGE_MEAN, IMAGE_STD);
    }

    protected void runTflite() {
      tfLite.run(input, labelProb);
    }

    protected void onRunTfliteDone() {
      Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));
      result.success(GetTopN(NUM_RESULTS, THRESHOLD));
    }
  }


  byte[] fetchArgmax(ByteBuffer output, List<Number> labelColors, String outputType) {
    Tensor outputTensor = tfLite.getOutputTensor(0);
    int outputBatchSize = outputTensor.shape()[0];
    assert outputBatchSize == 1;
    int outputHeight = outputTensor.shape()[1];
    int outputWidth = outputTensor.shape()[2];
    int outputChannels = outputTensor.shape()[3];

    Bitmap outputArgmax = null;
    byte[] outputBytes = new byte[outputWidth * outputHeight * 4];
    if (outputType.equals("png")) {
      outputArgmax = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
    }

    if (outputTensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          int maxIndex = 0;
          float maxValue = 0.0f;
          for (int c = 0; c < outputChannels; ++c) {
            float outputValue = output.getFloat();
            if (outputValue > maxValue) {
              maxIndex = c;
              maxValue = outputValue;
            }
          }
          int labelColor = labelColors.get(maxIndex).intValue();
          if (outputType.equals("png")) {
            outputArgmax.setPixel(j, i, labelColor);
          } else {
            setPixel(outputBytes, i * outputWidth + j, labelColor);
          }
        }
      }
    } else {
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          int maxIndex = 0;
          int maxValue = 0;
          for (int c = 0; c < outputChannels; ++c) {
            int outputValue = output.get();
            if (outputValue > maxValue) {
              maxIndex = c;
              maxValue = outputValue;
            }
          }
          int labelColor = labelColors.get(maxIndex).intValue();
          if (outputType.equals("png")) {
            outputArgmax.setPixel(j, i, labelColor);
          } else {
            setPixel(outputBytes, i * outputWidth + j, labelColor);
          }
        }
      }
    }
    if (outputType.equals("png")) {
      return compressPNG(outputArgmax);
    } else {
      return outputBytes;
    }
  }

  void setPixel(byte[] rgba, int index, long color) {
    rgba[index * 4] = (byte) ((color >> 16) & 0xFF);
    rgba[index * 4 + 1] = (byte) ((color >> 8) & 0xFF);
    rgba[index * 4 + 2] = (byte) (color & 0xFF);
    rgba[index * 4 + 3] = (byte) ((color >> 24) & 0xFF);
  }

  byte[] compressPNG(Bitmap bitmap) {
    // https://stackoverflow.com/questions/4989182/converting-java-bitmap-to-byte-array#4989543
    ByteArrayOutputStream stream = new ByteArrayOutputStream();
    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
    byte[] byteArray = stream.toByteArray();
    // bitmap.recycle();
    return byteArray;
  }

  

  void initPoseNet(Map<Integer, Object> outputMap) {
    if (partsIds.size() == 0) {
      for (int i = 0; i < partNames.length; ++i)
        partsIds.put(partNames[i], i);

      for (int i = 0; i < poseChain.length; ++i) {
        parentToChildEdges.add(partsIds.get(poseChain[i][1]));
        childToParentEdges.add(partsIds.get(poseChain[i][0]));
      }
    }

    for (int i = 0; i < tfLite.getOutputTensorCount(); i++) {
      int[] shape = tfLite.getOutputTensor(i).shape();
      float[][][][] output = new float[shape[0]][shape[1]][shape[2]][shape[3]];
      outputMap.put(i, output);
    }
  }

  

  PriorityQueue<Map<String, Object>> buildPartWithScoreQueue(float[][][] scores,
                                                             double threshold,
                                                             int localMaximumRadius) {
    PriorityQueue<Map<String, Object>> pq =
        new PriorityQueue<>(
            1,
            new Comparator<Map<String, Object>>() {
              @Override
              public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                return Float.compare((float) rhs.get("score"), (float) lhs.get("score"));
              }
            });

    for (int heatmapY = 0; heatmapY < scores.length; ++heatmapY) {
      for (int heatmapX = 0; heatmapX < scores[0].length; ++heatmapX) {
        for (int keypointId = 0; keypointId < scores[0][0].length; ++keypointId) {
          float score = sigmoid(scores[heatmapY][heatmapX][keypointId]);
          if (score < threshold) continue;

          if (scoreIsMaximumInLocalWindow(
              keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores)) {
            Map<String, Object> res = new HashMap<>();
            res.put("score", score);
            res.put("y", heatmapY);
            res.put("x", heatmapX);
            res.put("partId", keypointId);
            pq.add(res);
          }
        }
      }
    }

    return pq;
  }

  boolean scoreIsMaximumInLocalWindow(int keypointId,
                                      float score,
                                      int heatmapY,
                                      int heatmapX,
                                      int localMaximumRadius,
                                      float[][][] scores) {
    boolean localMaximum = true;
    int height = scores.length;
    int width = scores[0].length;

    int yStart = Math.max(heatmapY - localMaximumRadius, 0);
    int yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
    for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
      int xStart = Math.max(heatmapX - localMaximumRadius, 0);
      int xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
      for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
        if (sigmoid(scores[yCurrent][xCurrent][keypointId]) > score) {
          localMaximum = false;
          break;
        }
      }
      if (!localMaximum) {
        break;
      }
    }

    return localMaximum;
  }

  float[] getImageCoords(Map<String, Object> keypoint,
                         int outputStride,
                         int numParts,
                         float[][][] offsets) {
    int heatmapY = (int) keypoint.get("y");
    int heatmapX = (int) keypoint.get("x");
    int keypointId = (int) keypoint.get("partId");
    float offsetY = offsets[heatmapY][heatmapX][keypointId];
    float offsetX = offsets[heatmapY][heatmapX][keypointId + numParts];

    float y = heatmapY * outputStride + offsetY;
    float x = heatmapX * outputStride + offsetX;

    return new float[]{y, x};
  }

  boolean withinNmsRadiusOfCorrespondingPoint(List<Map<String, Object>> poses,
                                              float squaredNmsRadius,
                                              float y,
                                              float x,
                                              int keypointId) {
    for (Map<String, Object> pose : poses) {
      Map<Integer, Object> keypoints = (Map<Integer, Object>) pose.get("keypoints");
      Map<String, Object> correspondingKeypoint = (Map<String, Object>) keypoints.get(keypointId);
      float _x = (float) correspondingKeypoint.get("x") * inputSize - x;
      float _y = (float) correspondingKeypoint.get("y") * inputSize - y;
      float squaredDistance = _x * _x + _y * _y;
      if (squaredDistance <= squaredNmsRadius)
        return true;
    }

    return false;
  }

  Map<String, Object> traverseToTargetKeypoint(int edgeId,
                                               Map<String, Object> sourceKeypoint,
                                               int targetKeypointId,
                                               float[][][] scores,
                                               float[][][] offsets,
                                               int outputStride,
                                               float[][][] displacements) {
    int height = scores.length;
    int width = scores[0].length;
    int numKeypoints = scores[0][0].length;
    float sourceKeypointY = (float) sourceKeypoint.get("y") * inputSize;
    float sourceKeypointX = (float) sourceKeypoint.get("x") * inputSize;

    int[] sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypointY, sourceKeypointX,
        outputStride, height, width);

    float[] displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements);

    float[] displacedPoint = new float[]{
        sourceKeypointY + displacement[0],
        sourceKeypointX + displacement[1]
    };

    float[] targetKeypoint = displacedPoint;

    final int offsetRefineStep = 2;
    for (int i = 0; i < offsetRefineStep; i++) {
      int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1],
          outputStride, height, width);

      int targetKeypointY = targetKeypointIndices[0];
      int targetKeypointX = targetKeypointIndices[1];

      float offsetY = offsets[targetKeypointY][targetKeypointX][targetKeypointId];
      float offsetX = offsets[targetKeypointY][targetKeypointX][targetKeypointId + numKeypoints];

      targetKeypoint = new float[]{
          targetKeypointY * outputStride + offsetY,
          targetKeypointX * outputStride + offsetX
      };
    }

    int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1],
        outputStride, height, width);

    float score = sigmoid(scores[targetKeypointIndices[0]][targetKeypointIndices[1]][targetKeypointId]);

    Map<String, Object> keypoint = new HashMap<>();
    keypoint.put("score", score);
    keypoint.put("part", partNames[targetKeypointId]);
    keypoint.put("y", targetKeypoint[0] / inputSize);
    keypoint.put("x", targetKeypoint[1] / inputSize);

    return keypoint;
  }

  int[] getStridedIndexNearPoint(float _y, float _x, int outputStride, int height, int width) {
    int y_ = Math.round(_y / outputStride);
    int x_ = Math.round(_x / outputStride);
    int y = y_ < 0 ? 0 : y_ > height - 1 ? height - 1 : y_;
    int x = x_ < 0 ? 0 : x_ > width - 1 ? width - 1 : x_;
    return new int[]{y, x};
  }

  float[] getDisplacement(int edgeId, int[] keypoint, float[][][] displacements) {
    int numEdges = displacements[0][0].length / 2;
    int y = keypoint[0];
    int x = keypoint[1];
    return new float[]{displacements[y][x][edgeId], displacements[y][x][edgeId + numEdges]};
  }

  float getInstanceScore(Map<Integer, Map<String, Object>> keypoints, int numKeypoints) {
    float scores = 0;
    for (Map.Entry<Integer, Map<String, Object>> keypoint : keypoints.entrySet())
      scores += (float) keypoint.getValue().get("score");
    return scores / numKeypoints;
  }

  private float sigmoid(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  private static Matrix getTransformationMatrix(final int srcWidth,
                                                final int srcHeight,
                                                final int dstWidth,
                                                final int dstHeight,
                                                final boolean maintainAspectRatio) {
    final Matrix matrix = new Matrix();

    if (srcWidth != dstWidth || srcHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) srcWidth;
      final float scaleFactorY = dstHeight / (float) srcHeight;

      if (maintainAspectRatio) {
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    matrix.invert(new Matrix());
    return matrix;
  }

  private void close() {
    if (tfLite != null)
      tfLite.close();
    labels = null;
    labelProb = null;
  }
}
