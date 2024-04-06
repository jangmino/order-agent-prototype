//
//  ContentView.swift
//  order-agent-prototype
//
//  Created by 오장민 on 3/6/24.
//

import SwiftUI
import AVFoundation
import whisperllamalib

struct ContentView: View {
    @StateObject var whisperState: WhisperState
    @StateObject var llamaState: LlamaState
    
    @State private var multiLineText = "<start_of_turn>user\n너는 사용자가 입력한 주문 문장을 분석하는 에이전트이다. 주문으로부터 이를 구성하는 음식명, 옵션명, 수량을 차례대로 추출해야 한다.\n주문 문장:짜장면 한그릇하고요. 코카콜라 500ml 한병이요.<end_of_turn>\n<start_of_turn>model\n"
    
    init(isdummy:Bool = false) {
        _whisperState = StateObject(wrappedValue: WhisperState(isdummy: isdummy))
        _llamaState = StateObject(wrappedValue: LlamaState(isdummy: isdummy))
    }
    
    var body: some View {
        NavigationStack {
            VStack {
                HStack {
                    
                    Button(whisperState.isRecording ? "음성 주문 종료" : "음성 주문 시작", action: {
                        Task {
                            llamaState.generatedText = ""
                            await whisperState.toggleRecord()
                            if whisperState.transcripedText != "" {
                                await llamaState.complete(text: whisperState.transcripedText)
                            }
                        }
                    })
                    .buttonStyle(.bordered)
                    .font(.system(size: 14, weight: .semibold))
                    .disabled(!whisperState.canTranscribe)
                }
                .padding()
                
                VStack {
                    Text(whisperState.transcripedText)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.blue)
                    
                    Text(llamaState.generatedText)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.red)
                }
                .padding()
                
                ScrollView {
                    Text(whisperState.messageLog)
                        .font(.system(size: 12, weight: .semibold))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                
                ScrollView(.vertical, showsIndicators: true) {
                    Text(llamaState.messageLog)
                        .font(.system(size: 12, weight: .semibold))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .onTapGesture {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                }
//
//                
//                HStack {
//                    Button("Send") {
//                        sendText()
//                    }
//                    .buttonStyle(.bordered)
//                }
                
            }
            .navigationTitle("Order Agent Prototype")
            .padding()
        }.onAppear {
            Task {
                await whisperState.loadModels()
                await llamaState.loadModels()
            }
        }
    }
    
    func sendText() {
        Task {
//            await llamaState.complete(text: llm_prefix + multiLineText + "<end_of_turn>\n<start_of_turn>model\n")
            await llamaState.complete(text: whisperState.transcripedText)
//            multiLineText = ""
        }
    }
}

#Preview {
    ContentView(isdummy: true)
}
