// `hyprctl clients` and `hyprctl activewindow` with what the plugin does to each window:
// noScreenShare and noScreenShareCover (written by the core, src/hyprctl.rs).

#include "shim.hpp"

namespace nsc {

    namespace {

        using Handler = std::function<IPC::Socket1::SResponse(const IPC::Socket1::SRequest&)>;

        // Hyprland offers no way to add fields to a command's answer, and refuses a second
        // command of the same name. So the handlers of the two commands are wrapped, and put
        // back on unload. The handler found in a command is held by the wrapper alone (shared
        // by its copies), never here: it may be another plugin's wrapper, whose code goes with
        // that plugin. If that plugin puts back what it had found, it destroys our wrapper, and
        // the handler with it, while it is still loaded.
        struct SWrapped {
            SP<IPC::Socket1::SCommand>   cmd;
            std::weak_ptr<const Handler> original; // of the wrapper put in its place (SWrapper)
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

        // What stands in a command's handler while the plugin is loaded. Its own type tells it
        // apart from whatever another plugin may put there later. original is shared by every
        // copy of it, so whether it is still alive says whether something (the command, or
        // another plugin's wrapper around this one) still holds one.
        struct SWrapper {
            std::shared_ptr<const Handler> original;

            IPC::Socket1::SResponse operator()(const IPC::Socket1::SRequest& request) const {
                auto response = (*original)(request);
                if (auto* text = std::get_if<std::string>(&response.result))
                    *text = withFields(std::move(*text), request.format == IPC::Socket1::FORMAT_JSON);
                return response;
            }
        };

    } // namespace

    void wrapHyprctl() {
        const auto& sock = IPC::Socket1::sock();
        if (!sock)
            return;
        for (const auto& cmd : sock->m_commands) {
            if (!cmd || !cmd->handler || cmd->match != IPC::Socket1::COMMAND_MATCH_EXACT || (cmd->name != "clients" && cmd->name != "activewindow"))
                continue;
            auto original = std::make_shared<const Handler>(cmd->handler);
            g_wrapped.push_back({cmd, original});
            cmd->handler = SWrapper{.original = std::move(original)};
        }
    }

    // The wrappers are code of this plugin: none may stay behind. The original handler goes
    // back only while a wrapper still holds it:
    //   - the wrapper is still in place: the original replaces it;
    //   - another plugin wrapped it in turn and still holds it: the original goes back all the
    //     same. That plugin's wrapper is dropped with it, but kept it would call into this
    //     plugin once unloaded, and nothing can be taken out of the middle of the chain;
    //   - no wrapper is left (another plugin that had wrapped the command before this one put
    //     back what it had found): left as it is. The original went with the last wrapper,
    //     while that plugin was still loaded; nothing of it is left here to destroy, as its
    //     code may be gone by now.
    void unwrapHyprctl() {
        for (auto& w : g_wrapped) {
            // kept alive here: assigning the handler destroys the wrapper that holds it
            const auto original = w.original.lock();
            if (!original) {
                NSC_TRACE("hyprctl %s: replaced by someone else, left as it is\n", w.cmd->name.c_str());
                continue;
            }
            const auto* ours = w.cmd->handler.target<SWrapper>();
            if (!ours || ours->original != original)
                NSC_TRACE("hyprctl %s: wrapped again by someone else, its wrapper goes with this one\n", w.cmd->name.c_str());
            w.cmd->handler = *original;
        }
        g_wrapped.clear();
    }

} // namespace nsc
