using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveAccrosPlane : MonoBehaviour
{
    GameObject turtle;
    Animator animator;
    Vector3 _movement;
    float lerpDuration = 0.5f;
    float moveDuration = 3.5f;
    bool rotating;
    bool direction = true;
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
                animator.Play("Walk");
            }
            else
            {
                animator.Play("Idle");
            }
        }
        if(!rotating && !_walking)
        {
            rotating = true;
            StartCoroutine(Rotate180());
        }
        if (!_walking )
        {
            StartCoroutine(Move());
            turtle.transform.Translate(_movement);
        }
    }

    IEnumerator Rotate180()
    {
        rotating = true;
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
        rotating = false;
        direction = !direction;
    }

    IEnumerator Move()
    {
        rotating = true;
        float timeElapsed = 0;
        Quaternion startRotation = _movement.dot;
        Quaternion targetRotation = transform.rotation * Quaternion.Euler(0, 180, 0);
        while (timeElapsed < lerpDuration)
        {
            transform.rotation = Quaternion.Slerp(startRotation, targetRotation, timeElapsed / lerpDuration);
            timeElapsed += Time.deltaTime;
            yield return null;
        }
        transform.rotation = targetRotation;
        rotating = false;
        direction = !direction;
    }
}
