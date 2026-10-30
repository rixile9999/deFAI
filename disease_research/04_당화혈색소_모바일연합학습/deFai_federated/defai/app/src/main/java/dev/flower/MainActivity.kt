package dev.flower

import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import dev.flower.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.logging.Handler
import java.util.logging.LogRecord
import java.util.logging.Logger

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.logTextview.movementMethod = ScrollingMovementMethod()

        binding.startButton.setOnClickListener {
            val hba1c = binding.hba1cInput.text.toString().toDoubleOrNull()
            val bloodSugar = binding.bloodSugarInput.text.toString().toDoubleOrNull()
            val weight = binding.weightInput.text.toString().toDoubleOrNull()
            val host = binding.serverHostInput.text.toString().trim()
            val port = binding.serverPortInput.text.toString().toIntOrNull()

            if (hba1c == null || bloodSugar == null || weight == null ||
                !hba1c.isFinite() || !bloodSugar.isFinite() || !weight.isFinite() ||
                hba1c <= 0 || bloodSugar <= 0 || weight <= 0 || host.isBlank() ||
                port == null || port !in 1..65535) {
                Toast.makeText(this, "모든 값을 올바르게 입력해주세요.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            binding.logTextview.text = ""
            startFederatedLearning(hba1c, bloodSugar, weight, host, port)
        }

        // Set up a handler to redirect logs to the TextView
        val logHandler = TextViewLogHandler(binding.logTextview)
        val rootLogger = Logger.getLogger("")
        rootLogger.addHandler(logHandler)
    }

    private fun startFederatedLearning(hba1c: Double, bloodSugar: Double, weight: Double, host: String, port: Int) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val client = SimpleSingleLayerClient(
                    serverHost = host,
                    serverPort = port,
                    hba1c = hba1c,
                    bloodSugar = bloodSugar,
                    weight = weight
                )
                if (!client.checkServerHealth()) {
                    runOnUiThread {
                        binding.logTextview.append("❌ 서버에 연결할 수 없습니다!\n")
                        binding.logTextview.append("   서버가 실행 중인지 확인하세요.\n")
                    }
                    return@launch
                }
                client.startFederatedLearning(totalRounds = 3)

            } catch (e: Exception) {
                runOnUiThread {
                    binding.logTextview.append("❌ 클라이언트 실행 중 오류 발생: ${e.message}\n")
                }
            }
        }
    }

    // Custom log handler to append logs to a TextView
    class TextViewLogHandler(private val textView: TextView) : Handler() {
        override fun publish(record: LogRecord?) {
            record?.message?.let { msg ->
                val formattedMessage = "${record.level}: $msg\n"
                textView.post { textView.append(formattedMessage) }
            }
        }

        override fun flush() {}

        override fun close() {}
    }
}
