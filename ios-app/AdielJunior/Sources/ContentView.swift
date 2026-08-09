import SwiftUI
import WebKit
import AVFoundation

// MARK: - Adiel Junior - iOS Native HUD
// אפליקציית iOS אמיתית - לא Expo, עם WebView ל-HUD המשוכלל

struct ContentView: View {
    @StateObject private var viewModel = AdielViewModel()
    @State private var showMicPermissionAlert = false
    @State private var backendURL: String = UserDefaults.standard.string(forKey: "backendURL") ?? "http://192.168.1.100:8765"
    @State private var isHUDMode = true
    
    var body: some View {
        ZStack {
            // Background - Iron Man style
            LinearGradient(
                colors: [Color(red: 0.01, green: 0.04, blue: 0.12), Color(red: 0.03, green: 0.07, blue: 0.18)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            
            // Grid pattern
            GridPattern()
                .opacity(0.15)
                .ignoresSafeArea()
            
            VStack(spacing: 0) {
                // Top Bar
                TopBarView(viewModel: viewModel, backendURL: $backendURL)
                
                // Main Content
                if isHUDMode {
                    // HUD WebView - The sophisticated HUD
                    HUDWebView(url: getHUDURL(), viewModel: viewModel)
                        .clipShape(RoundedRectangle(cornerRadius: 20))
                        .padding(.horizontal, 12)
                        .shadow(color: .cyan.opacity(0.3), radius: 20)
                } else {
                    // Native iOS Controls
                    NativeControlsView(viewModel: viewModel)
                }
                
                // Bottom Controls
                BottomBarView(viewModel: viewModel, isHUDMode: $isHUDMode)
            }
        }
        .onAppear {
            viewModel.requestPermissions()
            viewModel.connectWebSocket(url: backendURL)
        }
        .alert("מיקרופון נדרש", isPresented: $showMicPermissionAlert) {
            Button("הגדרות") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            Button("ביטול", role: .cancel) {}
        } message: {
            Text("אדיאל צריכה גישה למיקרופון כדי להקשיב ל-'אדיאל ג'וניור'. אפשר גם לכתוב בצ'אט.")
        }
    }
    
    private func getHUDURL() -> URL {
        // אם יש backend, טען את ה-HUD המשוכלל מהשרת, אחרת טען לוקלי
        if let url = URL(string: "\(backendURL.replacingOccurrences(of: ":8765", with: ":3000"))/advanced_hud.html") {
            return url
        }
        // Fallback - bundled HUD
        if let localPath = Bundle.main.path(forResource: "advanced_hud", ofType: "html") {
            return URL(fileURLWithPath: localPath)
        }
        return URL(string: "about:blank")!
    }
}

// MARK: - Top Bar
struct TopBarView: View {
    @ObservedObject var viewModel: AdielViewModel
    @Binding var backendURL: String
    @State private var showSettings = false
    
    var body: some View {
        HStack {
            // Logo
            HStack(spacing: 10) {
                ZStack {
                    Circle()
                        .fill(LinearGradient(colors: [.cyan, .green], startPoint: .topLeading, endPoint: .bottomTrailing))
                        .frame(width: 32, height: 32)
                        .shadow(color: .cyan.opacity(0.6), radius: 10)
                    Text("AJ")
                        .font(.system(size: 11, weight: .black, design: .monospaced))
                        .foregroundColor(.black)
                }
                VStack(alignment: .leading, spacing: 1) {
                    Text("ADIEL JUNIOR")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .tracking(1.5)
                    Text("MARK 85 • iOS • קול מקורי")
                        .font(.system(size: 8, design: .monospaced))
                        .foregroundColor(.white.opacity(0.5))
                }
            }
            
            Spacer()
            
            // Status
            HStack(spacing: 8) {
                Circle()
                    .fill(viewModel.isConnected ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                    .shadow(color: viewModel.isConnected ? .green : .red, radius: 6)
                Text(viewModel.isConnected ? "CONNECTED" : "OFFLINE")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(viewModel.isConnected ? .green : .red)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Color.white.opacity(0.06))
            .clipShape(Capsule())
            .overlay(Capsule().stroke(Color.white.opacity(0.08), lineWidth: 1))
            
            Button(action: { showSettings.toggle() }) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.7))
                    .frame(width: 32, height: 32)
                    .background(Color.white.opacity(0.06))
                    .clipShape(Circle())
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            LinearGradient(colors: [Color.cyan.opacity(0.06), Color.clear], startPoint: .leading, endPoint: .trailing)
                .background(Color.black.opacity(0.3))
        )
        .overlay(Rectangle().frame(height: 1).foregroundColor(Color.white.opacity(0.06)), alignment: .bottom)
        .sheet(isPresented: $showSettings) {
            SettingsSheet(backendURL: $backendURL, viewModel: viewModel)
        }
    }
}

// MARK: - HUD WebView
struct HUDWebView: UIViewRepresentable {
    let url: URL
    @ObservedObject var viewModel: AdielViewModel
    
    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.mediaTypesRequiringUserActionForPlayback = []
        
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.backgroundColor = .clear
        webView.isOpaque = false
        webView.scrollView.backgroundColor = .clear
        
        // JS Bridge for native communication
        let contentController = config.userContentController
        contentController.add(context.coordinator, name: "adielNative")
        
        return webView
    }
    
    func updateUIView(_ webView: WKWebView, context: Context) {
        let request = URLRequest(url: url)
        webView.load(request)
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(viewModel: viewModel)
    }
    
    class Coordinator: NSObject, WKScriptMessageHandler {
        let viewModel: AdielViewModel
        
        init(viewModel: AdielViewModel) {
            self.viewModel = viewModel
        }
        
        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            // Bridge from Web HUD to Native
            if let body = message.body as? [String: Any] {
                print("[iOS Bridge] \(body)")
            }
        }
    }
}

// MARK: - Native Controls (fallback)
struct NativeControlsView: View {
    @ObservedObject var viewModel: AdielViewModel
    
    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Reactor
                ZStack {
                    Circle()
                        .stroke(Color.cyan.opacity(0.2), style: StrokeStyle(lineWidth: 1, dash: [4, 8]))
                        .frame(width: 140, height: 140)
                    
                    Circle()
                        .fill(RadialGradient(colors: [.cyan.opacity(0.8), .cyan.opacity(0.2), .clear], center: .center, startRadius: 10, endRadius: 60))
                        .frame(width: 80, height: 80)
                        .shadow(color: .cyan, radius: 20)
                    
                    Text("AJ")
                        .font(.system(size: 20, weight: .black, design: .monospaced))
                        .foregroundColor(.white)
                }
                .padding(.vertical, 20)
                
                // Conversation
                ForEach(viewModel.messages) { msg in
                    HStack {
                        if msg.role == .user { Spacer() }
                        VStack(alignment: msg.role == .user ? .trailing : .leading, spacing: 4) {
                            Text(msg.text)
                                .font(.system(size: 13))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(msg.role == .user ? Color.cyan.opacity(0.15) : Color.white.opacity(0.06))
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.08), lineWidth: 1))
                            Text(msg.time, style: .time)
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(.white.opacity(0.4))
                        }
                        if msg.role == .assistant { Spacer() }
                    }
                }
            }
            .padding()
        }
    }
}

// MARK: - Bottom Bar
struct BottomBarView: View {
    @ObservedObject var viewModel: AdielViewModel
    @Binding var isHUDMode: Bool
    @State private var inputText: String = ""
    
    var body: some View {
        VStack(spacing: 8) {
            // Visualizer
            HStack(spacing: 3) {
                ForEach(0..<12) { i in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(LinearGradient(colors: [.cyan, .green], startPoint: .bottom, endPoint: .top))
                        .frame(width: 3, height: viewModel.isListening ? CGFloat.random(in: 8...28) : 6)
                        .animation(.easeInOut(duration: 0.3).delay(Double(i) * 0.05), value: viewModel.isListening)
                }
            }
            .frame(height: 28)
            .opacity(viewModel.isListening || viewModel.isSpeaking ? 1 : 0)
            
            // Input
            HStack(spacing: 10) {
                HStack {
                    TextField("דבר עם אדיאל בעברית...", text: $inputText)
                        .font(.system(size: 14))
                        .onSubmit { send() }
                    
                    Button(action: { viewModel.isListening ? viewModel.stopListening() : viewModel.startListening() }) {
                        Image(systemName: viewModel.isListening ? "mic.slash.fill" : "mic.fill")
                            .foregroundColor(viewModel.isListening ? .red : .cyan)
                            .font(.system(size: 14, weight: .bold))
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.08), lineWidth: 1))
                
                Button(action: send) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 32))
                        .foregroundColor(.cyan)
                }
            }
            
            // Quick actions
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    QuickActionButton(title: "🎤 אדיאל ג'וניור", action: { viewModel.sendWakeWord() })
                    QuickActionButton(title: "🛒 קנה הכי זול", action: { viewModel.executeTask("תקנה לי אוזניות הכי זול") })
                    QuickActionButton(title: "✈️ חופשה", action: { viewModel.executeTask("תזמין טיסה מתל אביב ללונדון") })
                    QuickActionButton(title: "🧠 אמן", action: { viewModel.executeTask("תחקור על AI") })
                    QuickActionButton(title: "🎙️ מיקרופון", action: { /* show mic */ })
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color.black.opacity(0.4))
        .background(.ultraThinMaterial)
    }
    
    func send() {
        guard !inputText.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        viewModel.sendText(inputText)
        inputText = ""
    }
}

struct QuickActionButton: View {
    let title: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.white.opacity(0.06))
                .clipShape(Capsule())
                .overlay(Capsule().stroke(Color.white.opacity(0.08), lineWidth: 1))
        }
    }
}

// MARK: - ViewModel
class AdielViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = [
        ChatMessage(role: .assistant, text: "שלום בוס! אני אדיאל ג'וניור iOS - עם קול מקורי, מוח אמיתי מאפס, וסוכן-על. תגיד אדיאל ג'וניור!", time: Date())
    ]
    @Published var isConnected = false
    @Published var isListening = false
    @Published var isSpeaking = false
    @Published var statusText = "מאזינה למילת הפעלה..."
    
    private var webSocketTask: URLSessionWebSocketTask?
    private var audioRecorder: AVAudioRecorder?
    private var audioEngine = AVAudioEngine()
    
    struct ChatMessage: Identifiable {
        let id = UUID()
        let role: Role
        let text: String
        let time: Date
        enum Role { case user, assistant }
    }
    
    func requestPermissions() {
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            print("[iOS] Mic permission: \(granted)")
        }
    }
    
    func connectWebSocket(url: String) {
        let wsURLString = url.replacingOccurrences(of: "http://", with: "ws://")
                             .replacingOccurrences(of: "https://", with: "wss://") + "/ws"
        
        guard let wsURL = URL(string: wsURLString) else { return }
        
        let session = URLSession(configuration: .default)
        webSocketTask = session.webSocketTask(with: wsURL)
        webSocketTask?.resume()
        
        isConnected = true
        listenWebSocket()
    }
    
    private func listenWebSocket() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .success(let message):
                if case .string(let text) = message {
                    if let data = text.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        DispatchQueue.main.async {
                            self?.handleWSMessage(json)
                        }
                    }
                }
                self?.listenWebSocket()
            case .failure(let error):
                print("[WS] Error: \(error)")
                self?.isConnected = false
            }
        }
    }
    
    private func handleWSMessage(_ json: [String: Any]) {
        guard let type = json["type"] as? String else { return }
        
        switch type {
        case "brain_response", "assistant_speaking":
            if let text = json["assistant_text"] as? String ?? json["text"] as? String {
                messages.append(ChatMessage(role: .assistant, text: text, time: Date()))
                isSpeaking = true
                // TTS עם AVSpeechSynthesizer בעברית
                speak(text: text)
                DispatchQueue.main.asyncAfter(deadline: .now() + Double(text.count) * 0.08) {
                    self.isSpeaking = false
                }
            }
        case "wake_detected":
            isListening = true
            statusText = "כן בוס?"
        case "stt_result":
            if let text = json["text"] as? String, !text.isEmpty {
                messages.append(ChatMessage(role: .user, text: text, time: Date()))
            }
            isListening = false
        default:
            break
        }
    }
    
    func sendText(_ text: String) {
        messages.append(ChatMessage(role: .user, text: text, time: Date()))
        let msg = ["type": "text", "text": text, "with_screen": false] as [String : Any]
        sendWSMessage(msg)
    }
    
    func sendWakeWord() {
        let msg = ["type": "manual_wake"] as [String : Any]
        sendWSMessage(msg)
        isListening = true
    }
    
    func executeTask(_ task: String) {
        sendText(task)
        // גם כ-task
        Task {
            // HTTP ל-/tasks/execute
        }
    }
    
    private func sendWSMessage(_ dict: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: dict),
              let str = String(data: data, encoding: .utf8) else { return }
        webSocketTask?.send(.string(str)) { error in
            if let error = error { print("[WS] Send error: \(error)") }
        }
    }
    
    func startListening() {
        isListening = true
        // כאן היה AVAudioEngine + Speech framework לזיהוי "אדיאל ג'וניור" מקומי
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.isListening = false
        }
    }
    
    func stopListening() {
        isListening = false
        audioEngine.stop()
    }
    
    private func speak(text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "he-IL") ?? AVSpeechSynthesisVoice(language: "he")
        utterance.rate = 0.52
        let synthesizer = AVSpeechSynthesizer()
        synthesizer.speak(utterance)
    }
}

// MARK: - Helpers
struct GridPattern: View {
    var body: some View {
        Canvas { context, size in
            let gridSize: CGFloat = 40
            context.stroke(Path { path in
                for x in stride(from: 0, through: size.width, by: gridSize) {
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: size.height))
                }
                for y in stride(from: 0, through: size.height, by: gridSize) {
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: size.width, y: y))
                }
            }, with: .color(Color.cyan.opacity(0.04)), lineWidth: 1)
        }
    }
}

struct SettingsSheet: View {
    @Binding var backendURL: String
    @ObservedObject var viewModel: AdielViewModel
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section("Backend") {
                    TextField("http://192.168.1.100:8765", text: $backendURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Button("התחבר") {
                        UserDefaults.standard.set(backendURL, forKey: "backendURL")
                        viewModel.connectWebSocket(url: backendURL)
                        dismiss()
                    }
                }
                Section("מידע") {
                    Text("אדיאל ג'וניור iOS - MARK 85")
                    Text("קול מקורי • מוח אמיתי מאפס • סוכן-על")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("Backend: \(backendURL)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("הגדרות")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
