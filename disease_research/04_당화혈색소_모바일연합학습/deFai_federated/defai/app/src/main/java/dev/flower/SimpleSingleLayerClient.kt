package dev.flower

import com.google.gson.Gson
import com.google.gson.JsonParser
import kotlinx.coroutines.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.logging.Logger
import kotlin.random.Random

class SimpleSingleLayerClient(
    private val serverHost: String = "10.0.2.2",
    private val serverPort: Int = 8001,
    private val clientId: String = "single_layer_client_${Random.nextInt(1000, 9999)}",
    private val hba1c: Double,
    private val bloodSugar: Double,
    private val weight: Double
) {
    private val logger: Logger
    private val httpClient: OkHttpClient
    private val gson: Gson
    private val model: SingleLayerModel

    init {
        logger = Logger.getLogger(SimpleSingleLayerClient::class.java.name)
        logger.info("🔧 단일 레이어 클라이언트 초기화 시작")
        logger.info("   서버: $serverHost:$serverPort")
        logger.info("   클라이언트 ID: $clientId")

        httpClient = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(10, TimeUnit.SECONDS)
            .writeTimeout(10, TimeUnit.SECONDS)
            .build()
        logger.info("✅ HTTP 클라이언트 초기화 완료")

        gson = Gson()
        logger.info("✅ JSON 파서 초기화 완료")

        model = SingleLayerModel(clientId, bloodSugar, weight, hba1c)
        logger.info("✅ 단일 레이어 모델 초기화 완료")
        logger.info("🎉 클라이언트 초기화 완료")
    }

    fun startFederatedLearning(totalRounds: Int = 3) {
        logger.info("🚀 단일 레이어 연합학습 시작")
        logger.info("   서버: $serverHost:$serverPort")
        logger.info("   클라이언트 ID: $clientId")
        logger.info("   총 라운드: $totalRounds")

        for (round in 1..totalRounds) {
            logger.info("\n📊 라운드 $round/$totalRounds 시작")

            val (serverParams, serverRound) = getParametersFromServerWithRound()
            if (serverParams == null || serverRound == null) {
                logger.severe("❌ 서버로부터 파라미터를 받지 못했습니다.")
                break
            }
            logger.info("   [서버 라운드: $serverRound, 클라이언트 라운드: $round]")

            model.setParameters(serverParams)
            logger.info("✅ 서버 파라미터 적용 완료")

            val trainingResult = model.train(epochs = 2)
            logger.info("✅ 로컬 훈련 완료 - 손실: ${trainingResult.loss}")

            val submitResult = submitParametersToServer(trainingResult.parameters, serverRound)
            if (!submitResult) {
                logger.severe("❌ 서버로 파라미터 전송 실패")
                break
            }

            logger.info("⏳ 서버 자동 집계 대기...")
            var advanced = false
            for (attempt in 1..100) {
                val (_, observedRound) = getParametersFromServerWithRound()
                if (observedRound != null && observedRound > serverRound) {
                    advanced = true
                    break
                }
                Thread.sleep(200)
            }
            if (!advanced) {
                logger.severe("Round did not complete within the wait limit; check other clients.")
                break
            }
            logger.info("✅ 라운드 $round 완료")
            Thread.sleep(1000)
        }

        logger.info("\n🎉 연합학습 완료!")
    }

    private fun getParametersFromServerWithRound(): Pair<List<ParameterInfo>?, Int?> {
        try {
            val request = Request.Builder()
                .url("http://$serverHost:$serverPort/get_parameters")
                .get()
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    logger.severe("HTTP 오류: ${response.code}")
                    return Pair(null, null)
                }

                val responseBody = response.body?.string() ?: run {
                    logger.severe("HTTP 응답 바디가 비어있습니다.")
                    return Pair(null, null)
                }

                val jsonResponse = JsonParser.parseString(responseBody).asJsonObject
                val status = jsonResponse.get("status").asString
                val serverRound = jsonResponse.get("round").asInt

                if (status == "success") {
                    val parametersJson = jsonResponse.getAsJsonArray("parameters")
                    val parameters = mutableListOf<ParameterInfo>()
                    for (paramJson in parametersJson) {
                        val param = paramJson.asJsonObject
                        val dataArray = param.getAsJsonArray("data")
                        val flattenedData = mutableListOf<Double>()
                        val firstElement = dataArray[0]
                        if (firstElement.isJsonArray) {
                            for (row in dataArray) {
                                val rowArray = row.asJsonArray
                                for (element in rowArray) {
                                    flattenedData.add(element.asDouble)
                                }
                            }
                        } else {
                            for (element in dataArray) {
                                flattenedData.add(element.asDouble)
                            }
                        }
                        val parameterInfo = ParameterInfo(
                            layer = param.get("layer").asInt,
                            shape = param.getAsJsonArray("shape").map { it.asInt }.toIntArray(),
                            data = flattenedData.toDoubleArray(),
                            size = param.get("size").asInt
                        )
                        parameters.add(parameterInfo)
                    }
                    return Pair(parameters, serverRound)
                } else {
                    logger.severe("서버 응답 오류: $responseBody")
                    return Pair(null, serverRound)
                }
            }
        } catch (e: IOException) {
            logger.severe("파라미터 수신 실패: ${e.message}")
            return Pair(null, null)
        }
    }

    private fun submitParametersToServer(parameters: List<ParameterInfo>, serverRound: Int?): Boolean {
        try {
            val requestBodyJson = gson.toJson(
                mapOf(
                    "client_id" to clientId,
                    "round" to serverRound,
                    "parameters" to parameters.map { param ->
                        mapOf(
                            "layer" to param.layer,
                            "shape" to param.shape.toList(),
                            "data" to param.data.toList(),
                            "size" to param.size
                        )
                    }
                )
            )
            val requestBody = requestBodyJson.toRequestBody("application/json".toMediaType())

            val request = Request.Builder()
                .url("http://$serverHost:$serverPort/submit_parameters")
                .post(requestBody)
                .build()

            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    logger.severe("HTTP 오류: ${response.code}")
                    return false
                }
                val responseBody = response.body?.string() ?: run {
                    logger.severe("HTTP 응답 바디가 비어있습니다.")
                    return false
                }

                val jsonResponse = JsonParser.parseString(responseBody).asJsonObject
                val status = jsonResponse.get("status").asString

                if (status == "success") {
                    logger.info("📤 서버로 파라미터 전송 완료")
                    return true
                } else {
                    logger.severe("서버 응답 오류: $responseBody")
                    return false
                }
            }
        } catch (e: IOException) {
            logger.severe("파라미터 전송 실패: ${e.message}")
            return false
        }
    }

    fun checkServerHealth(): Boolean {
        try {
            val request = Request.Builder()
                .url("http://$serverHost:$serverPort/health")
                .get()
                .build()

            httpClient.newCall(request).execute().use { response ->
                return response.isSuccessful
            }
        } catch (e: IOException) {
            return false
        }
    }
}

data class ParameterInfo(
    val layer: Int,
    val shape: IntArray,
    val data: DoubleArray,
    val size: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        
        other as ParameterInfo
        
        if (layer != other.layer) return false
        if (!shape.contentEquals(other.shape)) return false
        if (!data.contentEquals(other.data)) return false
        if (size != other.size) return false
        
        return true
    }
    
    override fun hashCode(): Int {
        var result = layer
        result = 31 * result + shape.contentHashCode()
        result = 31 * result + data.contentHashCode()
        result = 31 * result + size
        return result
    }
}

data class TrainingResult(
    val loss: Double,
    val numExamples: Int,
    val parameters: List<ParameterInfo>
)

class SingleLayerModel(
    private val clientId: String = "default",
    bloodSugar: Double,
    weight: Double,
    hba1c: Double
) {
    private val trainingData: List<Triple<Double, Double, Double>>
    private var weights: DoubleArray
    private var bias: Double
    private var lastLoss: Double

    init {
        println("🔧 단일 레이어 모델 초기화 시작")
        println("   클라이언트 ID: $clientId")
        trainingData = listOf(Triple(bloodSugar, weight, hba1c))
        println("✅ 훈련 데이터 생성 완료: ${trainingData.size}개 샘플, 데이터: ${trainingData[0]}")
        weights = DoubleArray(2) { (Math.random() * 2 - 1) * 0.1 }
        bias = (Math.random() * 2 - 1) * 0.1
        lastLoss = 0.5
        println("✅ 모델 파라미터 초기화 완료:")
        println("   가중치: ${weights.contentToString()}")
        println("   편향: $bias")
        println("🎉 단일 레이어 모델 초기화 완료")
    }
    
    fun getParameters(): List<ParameterInfo> {
        return listOf(
            ParameterInfo(0, intArrayOf(2, 1), weights, weights.size),
            ParameterInfo(1, intArrayOf(1), doubleArrayOf(bias), 1)
        )
    }
    
    fun setParameters(parameters: List<ParameterInfo>) {
        if (parameters.size >= 2) {
            weights = parameters[0].data
            bias = parameters[1].data[0]
        }
    }
    
    fun train(epochs: Int): TrainingResult {
        println("🔄 클라이언트 '$clientId' 단일 레이어 훈련 시작 (${trainingData.size}개 샘플)")
        
        repeat(epochs) { epoch ->
            var totalLoss = 0.0
            var numProcessed = 0
            
            for ((bloodSugar, weight, targetHba1c) in trainingData) {
                val normalizedBloodSugar = (bloodSugar - 150.0) / 50.0
                val normalizedWeight = (weight - 75.0) / 15.0
                
                val input = doubleArrayOf(normalizedBloodSugar, normalizedWeight)
                val prediction = forwardPass(input)
                
                val loss = (prediction - targetHba1c) * (prediction - targetHba1c)
                totalLoss += loss
                numProcessed++
                
                val gradient = 2.0 * (prediction - targetHba1c)
                updateWeights(gradient, input, learningRate = 0.01)
            }
            
            lastLoss = totalLoss / numProcessed
            println("   에포크 ${epoch + 1}: 손실 = ${String.format("%.4f", lastLoss)}")
        }
        
        val numExamples = trainingData.size
        println("✅ 클라이언트 '$clientId' 훈련 완료 - 최종 손실: ${String.format("%.4f", lastLoss)}")
        
        val parameters = getParameters()
        return TrainingResult(lastLoss, numExamples, parameters)
    }
    
    private fun forwardPass(input: DoubleArray): Double {
        var output = bias
        for (i in input.indices) {
            output += weights[i] * input[i]
        }
        return output
    }
    
    private fun updateWeights(gradient: Double, input: DoubleArray, learningRate: Double) {
        for (i in weights.indices) {
            weights[i] -= learningRate * gradient * input[i]
        }
        bias -= learningRate * gradient
    }
}
