// `hyprctl clients` and `hyprctl activewindow` with what the plugin does to each window:
// noScreenShare and noScreenShareCover (written by the core, src/hyprctl.rs).

#include "shim.hpp"

namespace nsc {

    namespace {

        using Handler = std::function<IPC::Socket1::SResponse(const IPC::Socket1::SRequest&)>;

        // Hyprland offers no way to add fields to a command's answer, and refuses a second
        // command of the same name. So the handlers of the two commands are wrapped, and put
        // back on unload.
        struct SWrapped {
            SP<IPC::Socket1::SCommand> cmd;
            Handler                    original;
        };

        std::vector<SWrapped> g_wrapped;

        std::string withFields(std::string text, bool json) {
            pushSettings(); // the default cover's file is the core's to name
            const auto&                  windows = Desktop::windowState()->windows();
            std::vector<SRuleValues>     rules; // owns the paths the infos point to
            std::vector<nsc_window_info> infos;
            rules.reserve(windows.size());
            for (const auto& w : windows) {
                if (!w)
                    continue;
                const bool  hidden = w->m_ruleApplicator && w->m_ruleApplicator->noScreenShare().valueOrDefault();
                const auto& r      = rules.emplace_back(hidden ? ruleValuesFor(w) : SRuleValues{});
                const auto  kind   = coverKind(r);
                infos.push_back({
                    .address   = reinterpret_cast<uintptr_t>(w.get()),
                    .rule_path = r.path ? r.path->c_str() : nullptr,
                    .kind      = kind.kind,
                    .amount    = kind.amount,
                    .hidden    = hidden,
                });
            }
            std::string out(text.size() + infos.size() * 160, '\0');
            auto        n = nsc_annotate_clients(text.data(), text.size(), json, infos.data(), infos.size(), out.data(), out.size());
            if (n > out.size()) {
                out.resize(n);
                n = nsc_annotate_clients(text.data(), text.size(), json, infos.data(), infos.size(), out.data(), out.size());
            }
            if (n == 0 || n > out.size())
                return text;
            out.resize(n);
            return out;
        }

    } // namespace

    void wrapHyprctl() {
        const auto& sock = IPC::Socket1::sock();
        if (!sock)
            return;
        for (const auto& cmd : sock->m_commands) {
            if (!cmd || !cmd->handler || cmd->match != IPC::Socket1::COMMAND_MATCH_EXACT || (cmd->name != "clients" && cmd->name != "activewindow"))
                continue;
            g_wrapped.push_back({cmd, cmd->handler});
            cmd->handler = [original = cmd->handler](const IPC::Socket1::SRequest& request) {
                auto response = original(request);
                if (auto* text = std::get_if<std::string>(&response.result))
                    *text = withFields(std::move(*text), request.format == IPC::Socket1::FORMAT_JSON);
                return response;
            };
        }
    }

    // The wrappers are code of this plugin: none may stay behind.
    void unwrapHyprctl() {
        for (auto& w : g_wrapped)
            w.cmd->handler = std::move(w.original);
        g_wrapped.clear();
    }

} // namespace nsc
