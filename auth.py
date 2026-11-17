import os
import streamlit as st
from supabase import create_client

@st.cache_resource
def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

def auth_panel():
    client = get_supabase()
    if not client:
        st.caption("Demo mode: add Supabase URL and anon key to enable login.")
        return None

    if st.session_state.get("auth_user"):
        user = st.session_state["auth_user"]
        st.success(f"Signed in: {user.get('email', 'user')}")
        if st.button("Sign out"):
            st.session_state.pop("auth_user", None)
            st.session_state.pop("auth_session", None)
            st.rerun()
        return user

    mode = st.radio("Account action", ["Sign in", "Create account"])
    email = st.text_input("Email", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")
    if st.button(mode):
        try:
            if mode == "Create account":
                result = client.auth.sign_up({"email": email, "password": password})
            else:
                result = client.auth.sign_in_with_password({"email": email, "password": password})
            if result.user:
                user = {"id": result.user.id, "email": result.user.email}
                st.session_state["auth_user"] = user
                st.session_state["auth_session"] = result.session
                st.success("Authentication successful.")
                st.rerun()
        except Exception as exc:
            st.error(f"Authentication error: {exc}")
    return None
