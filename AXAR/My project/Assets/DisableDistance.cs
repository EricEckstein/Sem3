using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DisableDistance : MonoBehaviour
{
    Camera m_MainCamera;
    GameObject turtle;
    Animator animator;

    void Start()
    {
        m_MainCamera = Camera.main;
        turtle = GameObject.FindGameObjectWithTag("Turtle");
        animator = turtle.GetComponent<Animator>();
    }

    // Update is called once per frame
    void Update()
    {

        if(Vector3.Distance(turtle.transform.position, m_MainCamera.transform.position) > 2)
        {
            //turtle.SetActive(false);
        }
        else
        {
            turtle.SetActive(true);
        }

        if (Vector3.Distance(turtle.transform.position, m_MainCamera.transform.position) < 0.2)
        {
            animator.Play("Dead");
        }
    }
}
