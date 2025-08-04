import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function Redirect404() {
  const navigate = useNavigate();

  useEffect(() => {
    navigate("/404", { replace: true });
  }, [navigate]);

  return null;
}
