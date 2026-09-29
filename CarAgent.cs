using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Globalization;

public class CarAgent : MonoBehaviour
{
    [Header("Car Movement")]
    public float moveSpeed = 5.0f;
    public float turnSpeed = 90.0f;
    public Camera agentCamera;

    [Header("Route order - drag checkpoint colliders here in driving order")]
    public Collider[] checkpointOrder;

    private Rigidbody rb;
    private Vector3 startPosition;
    private Quaternion startRotation;

    // 0 = safe, 1 = crash, 2 = correct checkpoint, 3 = lap complete
    private volatile int status = 0;
    private int nextCheckpointIndex = 0;

    private volatile bool commandReceived = false;
    private volatile bool resetRequested = false;
    private volatile bool resetComplete = false;
    private volatile float targetSteering = 0f;
    private volatile float targetThrottle = 0f;
    private float timeSinceLastCommand = 0f;

    private TcpListener server;
    private TcpClient client;
    private NetworkStream stream;
    private Thread serverThread;
    private volatile bool isRunning = false;

    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private byte[] latestFrameBytes;
    private readonly object frameLock = new object();

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        startPosition = transform.position;
        startRotation = transform.rotation;

        renderTexture = new RenderTexture(240, 200, 24);
        renderTexture.Create();
        texture2D = new Texture2D(240, 200, TextureFormat.RGB24, false);

        if (agentCamera == null)
            UnityEngine.Debug.LogError("[CarAgent] Assign Agent Camera in the Inspector.");
        else
            agentCamera.targetTexture = renderTexture;

        isRunning = true;
        serverThread = new Thread(RunServer) { IsBackground = true };
        serverThread.Start();

        UnityEngine.Debug.Log("[CarAgent] Server running on port 9091.");
    }

    void Update()
    {
        // RESET arrives on the socket thread, but Unity objects must change on main thread.
        if (resetRequested)
        {
            ResetCar();
            CaptureCameraFrame();
            resetRequested = false;
            resetComplete = true;
        }
        else
        {
            CaptureCameraFrame();
        }

        // Report a fall as a crash. Python will request the reset after receiving it.
        if (transform.position.y < -1.0f)
            status = 1;

        if (commandReceived)
        {
            timeSinceLastCommand = 0f;
            commandReceived = false;
        }

        timeSinceLastCommand += Time.deltaTime;

        // During PPO optimisation Python does not send commands. Freeze the car safely.
        if (timeSinceLastCommand > 0.2f)
        {
            targetSteering = 0f;
            targetThrottle = 0f;

            if (rb != null)
            {
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }
        }
    }

    void FixedUpdate()
    {
        if (rb == null || status == 1)
            return;

        Vector3 movement = rb.rotation * Vector3.forward
            * targetThrottle * moveSpeed * Time.fixedDeltaTime;
        Quaternion rotation = rb.rotation * Quaternion.Euler(
            0f,
            targetSteering * turnSpeed * Time.fixedDeltaTime,
            0f
        );

        rb.MovePosition(rb.position + movement);
        rb.MoveRotation(rotation);
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Obstacle") ||
            collision.gameObject.CompareTag("OffTrack"))
        {
            status = 1;
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Obstacle") || other.CompareTag("OffTrack"))
        {
            status = 1;
            return;
        }

        if (!other.CompareTag("Checkpoint") || checkpointOrder == null ||
            checkpointOrder.Length == 0 || status != 0)
        {
            return;
        }

        // Only the expected checkpoint advances the route.
        if (nextCheckpointIndex < checkpointOrder.Length &&
            other == checkpointOrder[nextCheckpointIndex])
        {
            nextCheckpointIndex++;

            if (nextCheckpointIndex == checkpointOrder.Length)
            {
                status = 3;
                UnityEngine.Debug.Log("[CarAgent] Lap complete.");
            }
            else
            {
                status = 2;
                UnityEngine.Debug.Log(
                    "[CarAgent] Correct checkpoint. Next: " +
                    checkpointOrder[nextCheckpointIndex].gameObject.name
                );
            }
        }
    }

    private void ResetCar()
    {
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.position = startPosition + new Vector3(0f, 0.05f, 0f);
            rb.rotation = startRotation;
        }
        else
        {
            transform.position = startPosition + new Vector3(0f, 0.05f, 0f);
            transform.rotation = startRotation;
        }

        status = 0;
        nextCheckpointIndex = 0;
        targetSteering = 0f;
        targetThrottle = 0f;
        timeSinceLastCommand = 0f;
    }

    private void CaptureCameraFrame()
    {
        if (agentCamera == null || renderTexture == null || texture2D == null)
            return;

        if (!renderTexture.IsCreated())
            renderTexture.Create();

        RenderTexture previous = RenderTexture.active;

        try
        {
            agentCamera.targetTexture = renderTexture;
            agentCamera.Render();
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, 240, 200), 0, 0);
            texture2D.Apply(false);

            lock (frameLock)
            {
                latestFrameBytes = texture2D.EncodeToJPG(85);
            }
        }
        finally
        {
            RenderTexture.active = previous;
        }
    }

    private void RunServer()
    {
        try
        {
            server = new TcpListener(IPAddress.Any, 9091);
            server.Start();

            while (isRunning)
            {
                client = server.AcceptTcpClient();
                stream = client.GetStream();

                using (StreamReader reader = new StreamReader(
                    stream, Encoding.UTF8, false, 1024, true))
                {
                    string message;

                    while (isRunning && client.Connected &&
                           (message = reader.ReadLine()) != null)
                    {
                        if (message == "RESET")
                        {
                            resetComplete = false;
                            resetRequested = true;

                            while (isRunning && !resetComplete)
                                Thread.Sleep(1);
                        }
                        else
                        {
                            string[] parts = message.Split(',');

                            if (parts.Length == 2 &&
                                float.TryParse(parts[0], NumberStyles.Float,
                                    CultureInfo.InvariantCulture, out float steering) &&
                                float.TryParse(parts[1], NumberStyles.Float,
                                    CultureInfo.InvariantCulture, out float throttle))
                            {
                                targetSteering = Mathf.Clamp(steering, -1f, 1f);
                                targetThrottle = Mathf.Clamp01(throttle);
                                commandReceived = true;
                            }
                        }

                        SendState();
                    }
                }

                client.Close();
                client = null;
            }
        }
        catch (SocketException) { }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError("[CarAgent Socket Error] " + ex.Message);
        }
    }

    private void SendState()
    {
        byte[] frame;
        lock (frameLock)
        {
            frame = latestFrameBytes ?? Array.Empty<byte>();
        }

        int statusToSend = status;
        byte[] frameLength = BitConverter.GetBytes(frame.Length);

        stream.WriteByte((byte)statusToSend);
        stream.Write(frameLength, 0, frameLength.Length);

        if (frame.Length > 0)
            stream.Write(frame, 0, frame.Length);

        stream.Flush();

        // Events are reported once; Python resets immediately on crash or lap completion.
        if (statusToSend != 0 && status == statusToSend)
            status = 0;
    }

    void OnApplicationQuit()
    {
        isRunning = false;
        stream?.Close();
        client?.Close();
        server?.Stop();
    }
}
