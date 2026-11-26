"""Quest 3 input over USB-C.

The headset browser loads a WebXR page served by this process. With
`adb reverse tcp:8080 tcp:8080` the page is reached at http://localhost:8080,
and because `localhost` is a potentially-trustworthy origin it is a secure
context -- so `navigator.xr` is available without HTTPS, and no ngrok tunnel or
certificate is needed.
"""
