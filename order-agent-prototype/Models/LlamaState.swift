//
//  LlamaState.swift
//  prototype-order-genie
//
//  Created by a Developer on 3/7/24.
//

import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var generatedText = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0

    private var model_type: String
    private var isdummy: Bool = false
    private var llamaContext: LlamaContext?
    private var defaultModelUrl: URL? {
        switch model_type {
        case "phi":
            Bundle.main.url(forResource: "phi2-Q4_K_M", withExtension: "gguf", subdirectory: ".")
        case "gemma":
            Bundle.main.url(forResource: "gemma-2b-it-Q4_K_M", withExtension: "gguf", subdirectory: ".")
        case "mistral":
            Bundle.main.url(forResource: "mistral-7b-instruct-v0.2-Q4_K_M", withExtension: "gguf", subdirectory: ".")
        case "llama":
            Bundle.main.url(forResource: "llama-2-7b-chat-hf-Q4_K_M", withExtension: "gguf", subdirectory: ".")
        default:
            Bundle.main.url(forResource: "no-model", withExtension: "gguf", subdirectory: ".")
        }
//        Bundle.main.url(forResource: "gemma-2b-it-Q4_K_M", withExtension: "gguf", subdirectory: ".")
    }

    init(isdummy: Bool = false, model_type: String = "llama") {
        self.isdummy = isdummy
        self.model_type = model_type
//        if isdummy {
//            return
//        }
//        print("start...model: \(String(describing: defaultModelUrl))")
//        
//        loadModelsFromDisk()
//        loadDefaultModels()
    }
    
    func loadModels() async {
        guard let modelUrl = defaultModelUrl, !isdummy else {
            self.messageLog += "LLamaState: Coud not locate model\n"
            return
        }
        await withCheckedContinuation { continuation in
            // 백그라운드 스레드에서 비동기 작업을 수행
            print("start of initialize: \(#file)")
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let llamaContext = try LlamaContext.create_context(path: modelUrl.path())
                    DispatchQueue.main.async {
                        self.llamaContext = llamaContext
                        self.messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
                        continuation.resume()
                        print("end of initialize: \(#file)")
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.messageLog += "Could not locate model\n"
                        continuation.resume(throwing: error as! Never)
                    }
                }
            }
        }
    }


    func complete(text: String) async {
        guard let llamaContext else {
            return
        }
        
        var prompted_text: String
        
        switch model_type {
        case "phi":
            prompted_text = "Instruct:너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:\(text)\nOutput:"
        case "gemma":
            prompted_text = "<start_of_turn>user\n너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:\(text)<end_of_turn>\n<start_of_turn>model\n"
        case "mistral":
            prompted_text = "[INST] 너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:\(text) [/INST] "
        case "llama":
            prompted_text = "[INST] <<SYS>>\n너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n<</SYS>>\n\n주문 문장:\(text) [/INST] "
        default:
            prompted_text = "\(text)"
        }

        let t_start = DispatchTime.now().uptimeNanoseconds
        await llamaContext.completion_init(text: prompted_text)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        messageLog += "\(prompted_text)"
        
        generatedText = ""

        while await !llamaContext.is_eos {
            let result = await llamaContext.completion_loop()
            messageLog += "\(result)"
            generatedText += "\(result)"
        }

        let t_end = DispatchTime.now().uptimeNanoseconds
        let t_generation = Double(t_end - t_heat_end) / NS_PER_S
        let tokens_per_second = Double(await llamaContext.n_len) / t_generation

        await llamaContext.clear()
        messageLog += """
            \n
            Done
            Heat up took \(t_heat)s
            Generated \(tokens_per_second) t/s\n
            """
    }

}

