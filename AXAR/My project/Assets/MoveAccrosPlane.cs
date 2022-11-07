using System.Collections;
using UnityEngine;

public class MoveAccrosPlane : MonoBehaviour
{
    GameObject turtle;
    Animator animator;
    Vector3 _movement;
    float lerpDuration = 0.5f;
    float moveDuration = 3f;
    bool needRotate, needMove = false;
    bool _walking;

    // Start is called before the first frame update
    void Start()
    {
        turtle = GameObject.FindGameObjectWithTag("Turtle");
        animator = turtle.GetComponent<Animator>();
        _movement = new Vector3(0, 0, 0.001f);
        _walking = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            _walking = !_walking;
            if (_walking == true)
            {
                needMove = true;
                needRotate = false;
                animator.Play("Walk");
            }
            else
            {
                needMove = false;
                needRotate = false;
                animator.Play("Idle");
            }
        }
        if(needRotate)
        {
            needRotate = false;
            StartCoroutine(Rotate180());
        }
        if (needMove)
        {
            needMove = false;
            StartCoroutine(Move());
        }
    }

    IEnumerator Rotate180()
    {
        float timeElapsed = 0;
        Quaternion startRotation = transform.rotation;
        Quaternion targetRotation = transform.rotation * Quaternion.Euler(0, 180, 0);
        while (timeElapsed < lerpDuration)
        {
            transform.rotation = Quaternion.Slerp(startRotation, targetRotation, timeElapsed / lerpDuration);
            timeElapsed += Time.deltaTime;
            yield return null;
        }
        transform.rotation = targetRotation;
        needMove = true;
    }

    IEnumerator Move()
    {
        float timeElapsed = 0;
        while (timeElapsed < moveDuration)
        {
            transform.Translate(0,0,0.1f* Time.deltaTime);
            timeElapsed = timeElapsed +  Time.deltaTime;
            yield return null;
        }
        needRotate = true;
    }
}
