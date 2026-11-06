import UIKit

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        true
    }

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        let config = UISceneConfiguration(name: "Default", sessionRole: connectingSceneSession.role)
        config.delegateClass = SceneDelegate.self
        return config
    }
}

final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = scene as? UIWindowScene else { return }
        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = AssistViewController()
        window.makeKeyAndVisible()
        self.window = window
    }
}

/**
 * ビルドのときに入れた設定（API キーなど）。ios/Secrets.xcconfig（コミットしない）か、GitHub Actions のシークレットから
 * Info.plist に入る。キーはアプリに入るので、ビルドしたアプリを人に渡さないこと（TestFlight はチームの中だけで）。
 */
enum Secrets {
    static func value(_ key: String, default fallbackValue: String = "") -> String {
        let v = (Bundle.main.object(forInfoDictionaryKey: key) as? String)?
            .trimmingCharacters(in: CharacterSet(charactersIn: " \"'")) ?? ""
        // 設定がないと、ビルド設定の名前がそのまま残ることがある
        if v.isEmpty || v.hasPrefix("$(") { return fallbackValue }
        return v
    }

    /// key がなければ other（例: CLOUD_VISION_API_KEY がなければ GCP_API_KEY）。
    static func value(_ key: String, fallback other: String) -> String {
        let v = value(key)
        return v.isEmpty ? value(other) : v
    }
}
